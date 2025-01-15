use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 辅助函数：从 safetensors 中提取张量
        let extract_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).expect("张量不存在");
            let shape = tensor.shape().to_vec();
            let data: Vec<f32> = tensor
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("转换失败")))
                .collect();
            Tensor::new(data, &shape)
        };

        // 预分配存储每层参数的向量
        let n_layers = config.num_hidden_layers;
        let mut rms_att_w = Vec::with_capacity(n_layers);
        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut wo = Vec::with_capacity(n_layers);
        let mut rms_ffn_w = Vec::with_capacity(n_layers);
        let mut w_up = Vec::with_capacity(n_layers);
        let mut w_gate = Vec::with_capacity(n_layers);
        let mut w_down = Vec::with_capacity(n_layers);

        // 加载每层的必要参数
        for i in 0..n_layers {
            rms_att_w.push(extract_tensor(&format!(
                "model.layers.{i}.input_layernorm.weight"
            )));
            wq.push(extract_tensor(&format!(
                "model.layers.{i}.self_attn.q_proj.weight"
            )));
            wk.push(extract_tensor(&format!(
                "model.layers.{i}.self_attn.k_proj.weight"
            )));
            wv.push(extract_tensor(&format!(
                "model.layers.{i}.self_attn.v_proj.weight"
            )));
            wo.push(extract_tensor(&format!(
                "model.layers.{i}.self_attn.o_proj.weight"
            )));
            rms_ffn_w.push(extract_tensor(&format!(
                "model.layers.{i}.post_attention_layernorm.weight"
            )));
            w_up.push(extract_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")));
            w_gate.push(extract_tensor(&format!(
                "model.layers.{i}.mlp.gate_proj.weight"
            )));
            w_down.push(extract_tensor(&format!(
                "model.layers.{i}.mlp.down_proj.weight"
            )));
        }

        // 返回 LLamaParams 结构体实例
        LLamaParams {
            embedding_table: extract_tensor(if config.tie_word_embeddings {
                "lm_head.weight"
            } else {
                "model.embed_tokens.weight"
            }),
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w: extract_tensor("model.norm.weight"),
            lm_head: extract_tensor("lm_head.weight"),
            

        }
    }
}





        







