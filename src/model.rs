use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::model;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::NumType;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::arch::x86_64::*;
use std::convert::TryInto;
use std::error::Error as StdError;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use std::vec;
use tokenizers::Tokenizer;
pub struct Llama<T: NumType> {
    vocab: usize,                                // vocab size
    n_layers: usize,                             // number of layers
    n_q_h: usize,                                // number of heads for q
    n_kv_h: usize,                               // number of heads for k and v
    d: usize,                                    // dimension of hidden states
    dqkv: usize,                                 // length of a single q, k, or v vector
    di: usize,                                   // dimension of intermediate states
    eps: f32,                                    // epsilon for RMS normalization
    rope_theta: f32,                             // rope theta for rope initialization
    max_seq_len: usize,                          // maximum sequence length
    params: LLamaParams<T, T>,                   // trained weights of this model
    bos_token_id: u32,                           // start token id
    pub eos_token_id: u32,                       // end token id
    matrix_ops_count: std::cell::Cell<usize>,    // 原始实现计数器
    optimized_ops_count: std::cell::Cell<usize>, // 优化实现计数器
    matrix_time: std::cell::Cell<f64>,           // 矩阵计算总时间
    tokenizer: Tokenizer,
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
            matrix_ops_count: std::cell::Cell::new(0),
            optimized_ops_count: std::cell::Cell::new(0),
            matrix_time: std::cell::Cell::new(0.0),
            tokenizer: Tokenizer::from_file(
                (PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("models")
                    .join("chat"))
                .join("tokenizer.json"),
            )
            .unwrap(),
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let start_time = Instant::now();
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table); //获取文本的词向量

        for layer in 0..self.n_layers {
            //归一化
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::Operators::<f32>::matmul_parallel(
                q,
                0.,
                &hidden_states,
                &self.params.wq[layer],
                1.0,
                true,
                Some((&self.matrix_ops_count, &self.optimized_ops_count)),
            );
            OP::Operators::<f32>::matmul_parallel(
                k,
                0.,
                &hidden_states,
                &self.params.wk[layer],
                1.0,
                true,
                Some((&self.matrix_ops_count, &self.optimized_ops_count)),
            );
            OP::Operators::<f32>::matmul_parallel(
                v,
                0.,
                &hidden_states,
                &self.params.wv[layer],
                1.0,
                true,
                Some((&self.matrix_ops_count, &self.optimized_ops_count)),
            );
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // Call self_attention
            {
                // score = Q @ K.T / sqrt(dim)
                q.reshape(&vec![seq_len, self.n_q_h * self.dqkv]);
                // 获得的形状，应该为n_q_h * seq_len * total
                OP::vec_multi(
                    &mut att_scores,
                    q,
                    full_k,
                    1. / (self.dqkv as f32).sqrt(),
                    true,
                );
                OP::masked_softmax(&mut att_scores);
                // x = attn @ V
                // 这里需要用到权重乘法，即上一步得出的是权重，接下来每一个权重对应的V向量
                // vec_multi(&mut hidden_states, att_scores, &full_k, 1., false);
                OP::vec_multi_wight(&mut hidden_states, &att_scores, &full_v);
                // x shape 6*128,
                OP::matmul_transb(
                    &mut residual,
                    1.,
                    &hidden_states,
                    &self.params.wo[layer],
                    1.,
                )
            }
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        let total_time = start_time.elapsed().as_secs_f64();
        self.matrix_time.set(total_time);

        // 输出优化统计信息
        // println!(
        //     "优化统计: 优化实现次数={}, 基础实现次数={}, 时间={:.4}秒",
        //     self.optimized_ops_count.get(),
        //     self.matrix_ops_count.get(),
        //     self.matrix_time.get()
        // );

        // 重置计数器和计时器
        self.matrix_ops_count.set(0);
        self.optimized_ops_count.set(0);
        self.matrix_time.set(0.0);

        logits
    }
    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = Vec::new();
        let mut cache = self.new_cache();
        let mut input_ids = token_ids.to_vec();

        for _ in 0..max_len {
            let shape = vec![input_ids.len()];
            let input = Tensor::new(input_ids.clone(), &shape);

            let output = self.forward(&input, &mut cache);
            let next_token = OP::random_sample(&output, top_p, top_k, temperature);

            if next_token == self.eos_token_id {
                break;
            }

            result.push(next_token);
            input_ids = vec![next_token];
        }

        result
    }
    pub fn chat(
        &self,
        messages: &[(&str, &str)], // 每条消息是一个元组 (role, content)
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> String {
        // 构建对话模板
        let mut prompt = String::new();
        for (role, content) in messages {
            prompt.push_str(&format!("<|im_start|>{}", role));
            prompt.push_str("\n");
            prompt.push_str(content);
            prompt.push_str("<|im_end|>");
            prompt.push_str("\n");
        }
        prompt.push_str("<|im_start|>assistant\n");

        // 将 prompt 转换为 &str，然后编码为 token_ids
        let token_ids = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .unwrap()
            .get_ids()
            .to_vec();

        // 使用 generate 函数生成助手的回复
        let generated_token_ids = self.generate(&token_ids, max_len, top_p, top_k, temperature);

        // 将生成的 token_ids 转换回字符串
        let response = self.tokenizer.decode(&generated_token_ids, true).unwrap();

        response
    }
}

pub struct LlamaChat {
    pub llama: model::Llama<f32>, // 模型作为结构体成员
    pub tokenizer: Tokenizer,     // tokenizer作为结构体成员
}

impl LlamaChat {
    /// 初始化方法
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let llama = Llama::<f32>::from_safetensors(model_dir.as_ref());
        let tokenizer = Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json"))?;

        Ok(Self { llama, tokenizer })
    }

    pub fn chat(
        &self,
        messages: &[(&str, &str)],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Result<Vec<u32>, Box<dyn Error + Send + Sync + 'static>> {
        let mut prompt = String::new();
        for (role, content) in messages {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
        }
        prompt.push_str("<|im_start|>assistant\n");

        let encoding = self.tokenizer.encode(prompt, true)?;

        Ok(self.llama.generate(
            encoding
                .get_ids()
                .iter()
                .map(|&id| id as u32)
                .collect::<Vec<_>>()
                .as_slice(),
            max_len,
            top_p,
            top_k,
            temperature,
        ))
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
    wo: &Tensor<f32>,
) -> Tensor<f32> {
    todo!("sss")
}

pub fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    OP::rms_norm(hidden_states, residual, rms_w, eps);
    OP::matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);
    OP::matmul_transb(up, 0.0, hidden_states, w_up, 1.0);
    OP::swiglu(up, gate);
    OP::matmul_transb(residual, 1.0, up, w_down, 1.0);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
