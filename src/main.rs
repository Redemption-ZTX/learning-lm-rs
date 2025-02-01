use eframe::egui;
use std::env;
use std::path::PathBuf;
use tokenizers::Tokenizer;

mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

#[derive(PartialEq, Eq)]
enum GenerationMode {
    Continue,
    Chat,
}

struct MyApp {
    input_text: String,
    output_text: String,
    llama_chat: Option<model::LlamaChat>,
    mode: GenerationMode,
    chat_history: Vec<(String, String)>,
}

impl MyApp {
    fn new() -> Self {
        Self {
            input_text: String::new(),
            output_text: String::new(),
            llama_chat: None,
            mode: GenerationMode::Continue,
            chat_history: Vec::new(),
        }
    }

    fn build_chat_prompt(&self) -> Vec<(&str, &str)> {
        self.chat_history
            .iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
            .collect()
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading(
                    egui::RichText::new("LLAMA Chatbot")
                        .color(egui::Color32::from_rgb(100, 200, 255))
                        .size(24.0),
                );
                ui.label(
                    egui::RichText::new("by TianxiangZhao")
                        .color(egui::Color32::from_rgb(150, 150, 150))
                        .size(14.0),
                );
            });

            // 模式选择
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.mode, GenerationMode::Continue, "continue-write");
                ui.radio_value(&mut self.mode, GenerationMode::Chat, "conversation");
            });

            // 输入区域
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new("Input:").color(egui::Color32::from_rgb(200, 200, 200)),
                );
                ui.add(
                    egui::TextEdit::multiline(&mut self.input_text)
                        .text_color(egui::Color32::WHITE)
                        .frame(true)
                        .desired_width(f32::INFINITY)
                        .desired_rows(5),
                );
            });

            // 操作按钮
            if ui
                .add(
                    egui::Button::new(
                        egui::RichText::new("Generate")
                            .color(egui::Color32::WHITE)
                            .size(16.0),
                    )
                    .fill(egui::Color32::from_rgb(0, 122, 204)),
                )
                .clicked()
            {
                self.output_text.clear();

                // 延迟加载模型
                if self.llama_chat.is_none() {
                    let project_dir = env!("CARGO_MANIFEST_DIR");
                    let model_dir = PathBuf::from(project_dir).join("models").join("story");

                    match model::LlamaChat::new(&model_dir) {
                        Ok(llama_chat) => self.llama_chat = Some(llama_chat),
                        Err(e) => self.output_text = format!("模型加载失败: {}", e),
                    }
                }

                if let Some(llama_chat) = &self.llama_chat {
                    match self.mode {
                        GenerationMode::Continue => {
                            // 续写模式 - 直接使用输入文本
                            if let Ok(input_ids) =
                                llama_chat.tokenizer.encode(&*self.input_text, true)
                            {
                                let output_ids = llama_chat.llama.generate(
                                    &input_ids.get_ids().to_vec(),
                                    500, // max_len
                                    0.8, // top_p
                                    30,  // top_k
                                    1.0, // temperature
                                );
                                self.output_text = llama_chat
                                    .tokenizer
                                    .decode(&output_ids, true)
                                    .unwrap_or_else(|e| format!("解码失败: {}", e));
                            } else {
                                self.output_text = "输入编码失败".to_string();
                            }
                        }
                        GenerationMode::Chat => {
                            // 对话模式 - 构建对话格式prompt
                            if !self.input_text.is_empty() {
                                self.chat_history
                                    .push(("user".to_string(), self.input_text.clone()));
                            }

                            let messages = self.build_chat_prompt();
                            let mut prompt = String::new();
                            for (role, content) in messages {
                                prompt.push_str(&format!(
                                    "<|im_start|>{}\n{}<|im_end|>\n",
                                    role, content
                                ));
                            }
                            prompt.push_str("<|im_start|>assistant\n");

                            match llama_chat.tokenizer.encode(&*prompt, true) {
                                Ok(input_ids) => {
                                    let output_ids = llama_chat.llama.generate(
                                        &input_ids.get_ids().to_vec(),
                                        500, // max_len
                                        0.8, // top_p
                                        30,  // top_k
                                        1.0, // temperature
                                    );
                                    let decoded = llama_chat
                                        .tokenizer
                                        .decode(&output_ids, true)
                                        .unwrap_or_else(|e| format!("解码失败: {}", e));
                                    self.output_text = decoded.clone();
                                    self.chat_history.push(("assistant".to_string(), decoded));
                                    self.input_text.clear();
                                }
                                Err(_) => {
                                    self.output_text = "输入编码失败".to_string();
                                }
                            }
                        }
                    }
                }
            }

            // 显示聊天历史
            if self.mode == GenerationMode::Chat {
                ui.separator();
                ui.label("chat history:");
                for (role, content) in &self.chat_history {
                    ui.horizontal(|ui| {
                        ui.label(format!("{}:", role));
                        ui.text_edit_singleline(&mut format!("{}", content));
                    });
                }
            }

            // 输出区域
            ui.separator();
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new("Output:").color(egui::Color32::from_rgb(200, 200, 200)),
                );
                ui.add(
                    egui::TextEdit::multiline(&mut self.output_text)
                        .text_color(egui::Color32::WHITE)
                        .frame(true)
                        .desired_width(f32::INFINITY)
                        .desired_rows(10)
                        .interactive(false),
                );
            });
        });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([400.0, 300.0])
            .with_decorations(true),
        ..Default::default()
    };

    eframe::run_native(
        "LLAMA Chatbot - by TianxiangZhao",
        options,
        Box::new(|cc| {
            // 设置视觉样式
            cc.egui_ctx.set_visuals(egui::Visuals::dark());

            Box::new(MyApp::new())
        }),
    )?;
    Ok(())
}
