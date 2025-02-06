use half::prelude::*;
use num_traits::Float;
use rayon::prelude::*;
use std::arch::x86_64::*;
use std::usize; // 添加 f16 支持

use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let shape = if x.shape().len() == 1 {
        vec![1, x.shape()[0]]
    } else {
        x.shape().clone()
    };

    // 提取行数和列数
    let num_rows = shape[0];
    let num_cols = shape[1];

    // 检查 w 的形状
    assert!(
        w.shape().len() == 1 && w.shape()[0] == num_cols,
        "Weight tensor 'w' must be one-dimensional and match the number of columns in 'x'."
    );

    let x_data = x.data();
    let w_data = w.data();
    let y_data = unsafe { y.data_mut() };

    for row in 0..num_rows {
        let start_index = row * num_cols;
        let end_index = start_index + num_cols;

        // 计算当前行的平方和
        let sum_of_squares: f32 = x_data[start_index..end_index]
            .iter()
            .map(|&value| value * value)
            .sum();

        // 计算当前行的 RMS
        let rms = (sum_of_squares / num_cols as f32 + epsilon).sqrt();

        // 归一化并应用权重
        for col in 0..num_cols {
            let index = start_index + col;
            y_data[index] = w_data[col] * x_data[index] / rms;
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    // let len = y.size();
    // assert!(len == x.size());

    // let _y = unsafe { y.data_mut() };
    // let _x = x.data();

    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
    let len = y.size();
    assert_eq!(x.size(), len);

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    let _x = x.data();
    let silu_x = _x.iter().map(|&x| sigmoid(x) * x).collect::<Vec<_>>();
    let mut idx = 0;
    unsafe {
        let mut _y = y.data_mut();
        for elem in _y.iter_mut() {
            *elem = *elem * silu_x[idx];
            idx += 1;
        }
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let a_rows = a.shape()[0];
    let a_cols = a.shape()[1];
    let b_rows = b.shape()[0];
    let b_cols = b.shape()[1];
    let c_rows = c.shape()[0];
    let c_cols = c.shape()[1];

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    //确保 B 的形状是 (n, k) 并在操作中被视为 (k, n)
    assert_eq!(
        a_cols, b_cols,
        "A's columns must match B's columns (B's transposed row count)"
    );
    assert_eq!(c_rows, a_rows, "C's rows must match A's rows");
    assert_eq!(
        c_cols, b_rows,
        "C's columns must match B's rows (B's transposed column count)"
    );

    for i in 0..a_rows {
        for j in 0..b_rows {
            let mut sum = 0.0;
            for k in 0..a_cols {
                // 计算 A 的第 i 行与 B 的第 j 列的内积
                sum += a_data[i * a_cols + k] * b_data[j * b_cols + k];
            }
            c_data[i * c_cols + j] = beta * c_data[i * c_cols + j] + alpha * sum;
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}
pub fn add(
    output: &mut Tensor<f32>, // 目标张量，存储相加的结果
    input: &Tensor<f32>,      // 输入张量
    scale: f32,               // 输入张量的缩放因子
) {
    // 确保两个张量的形状相同
    assert_eq!(
        output.shape(),
        input.shape(),
        "Shapes of output and input must match"
    );

    // 获取张量的数据
    let output_data = unsafe { output.data_mut() }; // 使用 unsafe 块
    let input_data = input.data();

    // 逐元素相加
    for (out_val, in_val) in output_data.iter_mut().zip(input_data.iter()) {
        *out_val += scale * in_val;
    }
}

pub fn sample_top_p_top_k(
    logits: &[f32],   // 模型的输出 logits
    top_p: f32,       // top-p 采样的参数
    top_k: usize,     // top-k 采样的参数
    temperature: f32, // 温度参数
) -> usize {
    // Step 1: 应用温度参数
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&logit| (logit / temperature).exp())
        .collect();
    let sum_probs: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|prob| *prob /= sum_probs);

    // Step 2: Top-k 采样
    let mut indexed_probs: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(idx, &prob)| (idx, prob))
        .collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k_probs: Vec<(usize, f32)> = indexed_probs.into_iter().take(top_k).collect();

    // Step 3: Top-p 采样
    let mut cumulative_prob = 0.0;
    let mut selected_indices = Vec::new();
    for (idx, prob) in top_k_probs {
        cumulative_prob += prob;
        selected_indices.push(idx);
        if cumulative_prob >= top_p {
            break;
        }
    }

    // Step 4: 从选定的索引中随机采样
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let selected_index = rng.gen_range(0..selected_indices.len());
    selected_indices[selected_index]
}

pub fn vec_multi(c: &mut Tensor<f32>, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32, t: bool) {
    //判断c的长度是否大于2
    assert!(
        c.shape().len() > 2,
        "vec_multi of dimentions must be at least 2"
    );
    //a 用于切分数据
    assert!(a.shape().len() == 2, "vec_multi of dimensions must be 2");
    assert!(b.shape().len() == 2, "vec_multi of dimensions must be 2");
    let shape = c.shape();
    //获取矩阵的行列数
    let (row, column) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    //获取n_q_h用于分组,这是c出去最后两个维度之外所有维度的乘积,
    //最后两个维度是数据，前面的维度是每个头所占的dim与头个数的乘积
    let q_head_len = shape[..shape.len() - 2].iter().product::<usize>();
    //确定qk的倍数关系,一般是多个q对应一个k
    let q_k_reflect = a.shape()[1] / b.shape()[1];
    let vec_len = a.shape()[1] / q_head_len; //这是每一个头的维度，一共有a.shape[1]的数据，除以一共的头数，就是维度
    let a_data = a.data();
    //用于获取q_head需要跳过的数值
    let a_skip = a.shape()[1];
    let b_data = b.data();
    //用于获取k_head需要跳过的数值
    let b_skip = b.shape()[1];
    let data = unsafe { c.data_mut() };

    data.fill(0.);
    let mut c_data_offset = 0;
    if t {
        //用于分组计算，每个输入，在每个请求头下的vjiv
        for i in 0..q_head_len {
            //计算每一个输入，在一个请求头下的total中的所有v
            for j in 0..row {
                //临时q_head值，j*q+skip用于跳过多头i*16用于跳过单头
                let a_tmp =
                    &a_data[(i * vec_len + j * a_skip)..(i * vec_len + j * a_skip) + vec_len];
                // 计算单一v
                for k in 0..column {
                    let b_tmp = &b_data[(k * b_skip + (i / q_k_reflect) * vec_len)
                        ..(k * b_skip + (i / q_k_reflect) * vec_len) + vec_len];
                    data[c_data_offset] = a_tmp
                        .iter()
                        .zip(b_tmp.iter())
                        .fold(0., |tmp, (a_val, b_val)| tmp + a_val * b_val)
                        * alpha;
                    c_data_offset += 1;
                }
            }
        }
    }
}
// 只用于得分计算
// a代表所处理的权重,b代表所要乘的向量
pub fn vec_multi_wight(c: &mut Tensor<f32>, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert!(
        b.shape().len() == 2,
        "matmul_transb of dimensions must be at least 2"
    );
    assert!(
        a.shape().len() == 4,
        "matmul_transb of dimensions must be  4 是att_scores)"
    );
    let q_header_len = a.shape()[..a.shape().len() - 2].iter().product::<usize>();
    let shape = a.shape();
    // 获取矩阵的行列数
    let (row, column) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    // 获取计算向量的长度
    let vec_len = b.shape()[1] / a.shape()[0];
    // 确认a，b需要的对应关系,默认a的长度大于b的长度
    let n_groups = a.shape()[1];
    let b_column = b.shape()[1];
    let mut data = unsafe { c.data_mut() };
    // 清理脏数据
    data.fill(0.);
    for i in 0..q_header_len {
        // 获取当前q下的的全部注意力
        let a_data = &a.data()[i * row * column..(i + 1) * row * column];
        // 循环计算每个当前q下，每个输入的v权重
        for c_i in 0..row {
            // 用于标记当前计算到那一列
            let mut b_data_row_offset = 0;
            let tmp_c_offset = n_groups * b_column * c_i + i * vec_len;
            // 获取c存储当先向量的位置，
            let tmp_c = &mut data[tmp_c_offset..tmp_c_offset + vec_len];
            // 获取一个输入的全部注意力
            a_data[c_i * column..(c_i + 1) * column]
                .iter()
                .for_each(|tmp| {
                    // 获取q，对应的v b_data_row_offset*b_column表示要跳过的input
                    // (q_header_len/n_groups)*vec_len 表示q对应的v
                    let tmp_offset = b_data_row_offset * b_column + (i / n_groups) * vec_len;
                    let b_data = &b.data()[tmp_offset..tmp_offset + vec_len];
                    b_data.iter().zip(tmp_c.iter_mut()).for_each(|(t_b, t_c)| {
                        *t_c += t_b * tmp;
                    });
                    // 进行偏移
                    b_data_row_offset += 1;
                });
        }
    }
}
// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

// 添加 Operators 结构体定义
pub struct Operators<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync + Default + 'static> Operators<T> {
    // 添加辅助函数：检查是否适合使用 FP16
    fn is_safe_for_fp16(values: &[T]) -> bool {
        let max_abs = values
            .iter()
            .map(|x| x.to_f32().unwrap().abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        max_abs < 65504.0 && max_abs > 6.1e-5
    }

    #[inline]
    pub fn matmul(a: &[T], b: &[T], c: &mut [T], m: usize, k: usize, n: usize) {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // 对于f32类型使用AVX2优化
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    matmul_f32_avx2(
                        a.as_ptr() as *const f32,
                        b.as_ptr() as *const f32,
                        c.as_mut_ptr() as *mut f32,
                        m,
                        k,
                        n,
                    );
                    return;
                }
            }
        }
        // 回退到普通实现
        Self::matmul_fallback(a, b, c, m, k, n);
    }

    #[inline]
    fn matmul_fallback(a: &[T], b: &[T], c: &mut [T], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..k {
                    sum = sum + a[i * k + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    #[inline]
    pub fn matmul_parallel(
        c: &mut Tensor<T>,
        beta: T,
        a: &Tensor<T>,
        b: &Tensor<T>,
        alpha: T,
        transpose_b: bool,
        counter: Option<(&std::cell::Cell<usize>, &std::cell::Cell<usize>)>,
    ) {
        let a_rows = a.shape()[0];
        let a_cols = a.shape()[1];
        let (b_rows, b_cols) = if transpose_b {
            (b.shape()[0], b.shape()[1])
        } else {
            (b.shape()[1], b.shape()[0])
        };
        let c_rows = c.shape()[0];
        let c_cols = c.shape()[1];

        // 检查矩阵维度是否匹配
        if transpose_b {
            assert_eq!(a_cols, b_cols, "A's columns must match B's columns");
        } else {
            assert_eq!(a_cols, b_rows, "A's columns must match B's rows");
        }
        assert_eq!(c_rows, a_rows, "C's rows must match A's rows");
        assert_eq!(
            c_cols,
            if transpose_b { b_rows } else { b_cols },
            "C's columns must match B's dimension"
        );

        let matrix_size = a_rows * b_rows * a_cols;
        let parallel_threshold = T::from(10_000_usize).unwrap();

        if T::from(matrix_size).unwrap() > parallel_threshold {
            // 检查是否适合使用 FP16
            let use_fp16 = Self::is_safe_for_fp16(a.data()) && Self::is_safe_for_fp16(b.data());

            if use_fp16 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                // 更新优化计数器
                if let Some((_, opt_counter)) = counter {
                    opt_counter.set(opt_counter.get() + 1);
                }

                // 转换为 FP16
                let a_fp16 = Self::convert_to_fp16(a.data());
                let b_fp16 = Self::convert_to_fp16(b.data());
                let mut results_fp32 = vec![0.0f32; c_rows * c_cols];
                let c_data: Vec<f32> = c.data().iter().map(|x| x.to_f32().unwrap()).collect();

                // 并行计算（在 FP32 中进行）
                results_fp32
                    .par_chunks_mut(c_cols)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        for j in 0..c_cols {
                            let mut sum_fp32 = 0.0f32;
                            for k in 0..a_cols {
                                let b_idx = if transpose_b {
                                    j * b_cols + k
                                } else {
                                    k * b_cols + j
                                };
                                sum_fp32 +=
                                    a_fp16[i * a_cols + k].to_f32() * b_fp16[b_idx].to_f32();
                            }
                            chunk[j] = beta.to_f32().unwrap() * c_data[i * c_cols + j]
                                + alpha.to_f32().unwrap() * sum_fp32;
                        }
                    });

                // 转换回原始类型
                unsafe {
                    let c_data = c.data_mut();
                    for (i, &val) in results_fp32.iter().enumerate() {
                        c_data[i] = T::from(val).unwrap();
                    }
                }
            } else {
                // 使用原始并行实现
                let a_data = a.data();
                let b_data = b.data();
                let c_data = unsafe { c.data_mut() };

                let mut results = vec![T::zero(); c_rows * c_cols];
                results
                    .par_chunks_mut(c_cols)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        for j in 0..c_cols {
                            let mut sum = T::zero();
                            for k in 0..a_cols {
                                let b_idx = if transpose_b {
                                    j * b_cols + k
                                } else {
                                    k * b_cols + j
                                };
                                sum = sum + a_data[i * a_cols + k] * b_data[b_idx];
                            }
                            chunk[j] = beta * c_data[i * c_cols + j] + alpha * sum;
                        }
                    });

                c_data.copy_from_slice(&results);
            }
        } else {
            // 更新基础实现计数器（串行实现）
            if let Some((basic_counter, _)) = counter {
                basic_counter.set(basic_counter.get() + 1);
            }

            // 串行实现
            unsafe {
                let c_data = c.data_mut();
                let a_data = a.data();
                let b_data = b.data();

                for i in 0..c_rows {
                    for j in 0..c_cols {
                        let mut sum = T::zero();
                        for k in 0..a_cols {
                            let b_idx = if transpose_b {
                                j * b_cols + k
                            } else {
                                k * b_cols + j
                            };
                            sum = sum + a_data[i * a_cols + k] * b_data[b_idx];
                        }
                        c_data[i * c_cols + j] = beta * c_data[i * c_cols + j] + alpha * sum;
                    }
                }
            }
        }
    }

    // 辅助函数：转换为 FP16
    fn convert_to_fp16(data: &[T]) -> Vec<f16> {
        data.iter()
            .map(|&x| f16::from_f32(x.to_f32().unwrap()))
            .collect()
    }
}

#[target_feature(enable = "avx2")]
unsafe fn matmul_f32_avx2(a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in (0..n).step_by(8) {
            let mut sum = _mm256_setzero_ps();
            for p in 0..k {
                let a_val = _mm256_set1_ps(*a.add(i * k + p));
                let b_val = _mm256_loadu_ps(b.add(p * n + j));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val, b_val));
            }
            _mm256_storeu_ps(c.add(i * n + j), sum);
        }
    }
}
