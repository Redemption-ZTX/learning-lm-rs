mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use chrono::{DateTime, Local};
use eframe::egui;
use model::LlamaChat;
use std::env;
use std::path::PathBuf;
use std::time::Instant;
use uuid::Uuid;

#[derive(PartialEq, Eq)]
enum GenerationMode {
    Continue,
    Chat,
}

#[derive(Clone)]
struct ChatSession {
    id: String,
    name: String,
    history: Vec<(String, String)>,
    created_at: DateTime<Local>,
}

impl ChatSession {
    fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            history: Vec::new(),
            created_at: Local::now(),
        }
    }
}

struct MyApp {
    input_text: String,
    output_text: String,
    chat_llama: Option<LlamaChat>,
    story_llama: Option<LlamaChat>,
    mode: GenerationMode,
    sessions: Vec<ChatSession>,
    current_session_id: Option<String>,
    session_name_input: String,
}

impl MyApp {
    fn new() -> Self {
        Self {
            input_text: String::new(),
            output_text: String::new(),
            chat_llama: None,
            story_llama: None,
            mode: GenerationMode::Chat,
            sessions: Vec::new(),
            current_session_id: None,
            session_name_input: String::new(),
        }
    }

    fn current_session(&self) -> Option<&ChatSession> {
        self.current_session_id
            .as_ref()
            .and_then(|id| self.sessions.iter().find(|s| s.id == *id))
    }

    fn current_session_mut(&mut self) -> Option<&mut ChatSession> {
        let id = self.current_session_id.clone()?;
        self.sessions.iter_mut().find(|s| s.id == id)
    }

    fn handle_chat_generation(&mut self) {
        // 先获取输入文本的拷贝
        let input_text = self.input_text.clone();
        if input_text.is_empty() {
            return;
        }

        // 确保模型已加载
        if self.chat_llama.is_none() {
            let project_dir = env!("CARGO_MANIFEST_DIR");
            let model_dir = PathBuf::from(project_dir).join("models").join("chat");
            match LlamaChat::new(&model_dir) {
                Ok(llama) => self.chat_llama = Some(llama),
                Err(e) => {
                    self.output_text = format!("Chat model loading failed: {}", e);
                    return;
                }
            }
        }

        // 生成回复
        if let Some(llama) = &self.chat_llama {
            let input = vec![("user", input_text.as_str())];
            match llama.chat(&input, 200, 0.9, 3, 0.8) {
                Ok(output_ids) => {
                    if let Ok(response) = llama.tokenizer.decode(&output_ids, true) {
                        // 更新会话历史
                        if let Some(session) = self.current_session_mut() {
                            session.history.push(("user".to_string(), input_text));
                            session
                                .history
                                .push(("assistant".to_string(), response.clone()));
                        }
                        self.output_text = response;
                    }
                }
                Err(e) => {
                    self.output_text = format!("Generation failed: {}", e);
                }
            }
        }

        // 清除输入
        self.input_text.clear();
    }

    fn delete_session(&mut self, session_id: String) {
        if let Some(current_id) = &self.current_session_id {
            if current_id == &session_id {
                self.current_session_id = None;
            }
        }
        self.sessions.retain(|s| s.id != session_id);
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Title
            ui.vertical_centered(|ui| {
                ui.heading(
                    egui::RichText::new("LLAMA AI BY TianxiangZhao")
                        .color(egui::Color32::from_rgb(100, 200, 255))
                        .size(24.0),
                );
            });

            // Sessions management
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mode, GenerationMode::Chat, "Chat Mode");
                ui.selectable_value(&mut self.mode, GenerationMode::Continue, "Story Mode");

                if self.mode == GenerationMode::Chat {
                    // New session input
                    ui.text_edit_singleline(&mut self.session_name_input);
                    if ui.button("New Session").clicked() && !self.session_name_input.is_empty() {
                        let new_session = ChatSession::new(self.session_name_input.clone());
                        self.current_session_id = Some(new_session.id.clone());
                        self.sessions.push(new_session);
                        self.session_name_input.clear();
                    }
                }
            });

            // Session selector
            if self.mode == GenerationMode::Chat {
                let mut session_to_delete = None;
                ui.horizontal_wrapped(|ui| {
                    for session in &self.sessions {
                        let is_selected = self
                            .current_session_id
                            .as_ref()
                            .map_or(false, |id| id == &session.id);

                        let text = format!(
                            "{} ({})",
                            session.name,
                            session.created_at.format("%Y-%m-%d %H:%M")
                        );

                        if ui.selectable_label(is_selected, text).clicked() {
                            self.current_session_id = Some(session.id.clone());
                        }

                        if ui.button("🗑").clicked() {
                            session_to_delete = Some(session.id.clone());
                        }
                    }
                });

                // 在循环外处理删除操作
                if let Some(id) = session_to_delete {
                    self.delete_session(id);
                }
            }

            // / Input and generation area
            if self.mode == GenerationMode::Chat && self.current_session().is_none() {
                ui.label("Please create or select a session to start chatting");
            } else {
                // Input area
                ui.vertical(|ui| {
                    ui.label("Input:");
                    ui.text_edit_multiline(&mut self.input_text);
                });

                // Generate button
                if ui.button("Generate").clicked() {
                    let start_time = Instant::now();
                    match self.mode {
                        GenerationMode::Chat => self.handle_chat_generation(),
                        GenerationMode::Continue => {
                            // Lazy load story model
                            if self.story_llama.is_none() {
                                let project_dir = env!("CARGO_MANIFEST_DIR");
                                let model_dir =
                                    PathBuf::from(project_dir).join("models").join("story");
                                match LlamaChat::new(&model_dir) {
                                    Ok(llama) => self.story_llama = Some(llama),
                                    Err(e) => {
                                        self.output_text =
                                            format!("Story model loading failed: {}", e)
                                    }
                                }
                            }

                            if let Some(llama) = &self.story_llama {
                                let input = vec![("user", self.input_text.as_str())];
                                match llama.chat(&input, 200, 0.9, 3, 0.8) {
                                    Ok(output_ids) => {
                                        if let Ok(response) =
                                            llama.tokenizer.decode(&output_ids, true)
                                        {
                                            self.output_text = response;
                                        }
                                    }
                                    Err(e) => {
                                        self.output_text = format!("Generation failed: {}", e)
                                    }
                                }
                            }
                        }
                    }

                    let total_time = start_time.elapsed().as_secs_f64();
                    println!("\nTotal time: {:.4} seconds", total_time);
                }

                // Undo button (history rollback)
                if self.mode == GenerationMode::Chat {
                    if let Some(session) = self.current_session() {
                        if !session.history.is_empty() {
                            if ui.button("Undo Last Message").clicked() {
                                if let Some(session) = self.current_session_mut() {
                                    session.history.pop(); // Remove assistant's message
                                    session.history.pop(); // Remove user's message
                                }
                            }
                        }
                    }
                }

                /// Chat history displayl
                if self.mode == GenerationMode::Chat {
                    if let Some(session) = self.current_session() {
                        ui.separator();
                        ui.label(format!("Chat History - {}", session.name));
                        for (role, content) in &session.history {
                            ui.horizontal(|ui| {
                                ui.label(format!("{}:", role));
                                ui.label(content);
                            });
                        }
                    }
                }
            }

            // Output area
            ui.separator();
            ui.vertical(|ui| {
                ui.label("Output:");
                ui.add(
                    egui::TextEdit::multiline(&mut self.output_text)
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
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    eframe::run_native(
        "LLAMA AI",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Box::new(MyApp::new())
        }),
    )?;

    Ok(())
}
