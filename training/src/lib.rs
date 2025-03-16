mod training;
mod training_config;
mod training_history;

pub use training::Trainer;
pub use training_config::TrainingConfig;
pub use training_history::TrainingHistory;

pub mod prelude {
    pub use crate::Trainer;
    pub use crate::TrainingConfig;
    pub use crate::TrainingHistory;
}
