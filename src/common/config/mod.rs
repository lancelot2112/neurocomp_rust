pub mod constants;

use once_cell::sync::OnceCell;

#[derive(Clone, Debug)]
pub struct KernelDefaults {
    pub temperature_min: i16,
    pub temperature_max: i16,
    pub temperature_initial: i16,
}

#[derive(Clone, Debug)]
pub struct RenderDefaults {
    pub cols: usize,
    pub cell_bits: usize,
}

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub kernel: KernelDefaults,
    pub render: RenderDefaults,
}

impl Default for AppConfig {
    fn default() -> Self {
        use constants::*;
        Self {
            kernel: KernelDefaults {
                temperature_min: KERNEL_TEMP_MIN,
                temperature_max: KERNEL_TEMP_MAX,
                temperature_initial: KERNEL_TEMP_INITIAL,
            },
            render: RenderDefaults {
                cols: DEFAULT_RENDER_COLS,
                cell_bits: DEFAULT_RENDER_CELL_BITS,
            },
        }
    }
}

static GLOBAL_CONFIG: OnceCell<AppConfig> = OnceCell::new();

/// Initialize once (optional; if not called, defaults are used).
pub fn init(config: AppConfig) {
    let _ = GLOBAL_CONFIG.set(config);
}

/// Get the global config (lazily defaults if not set).
pub fn config() -> &'static AppConfig {
    GLOBAL_CONFIG.get_or_init(AppConfig::default)
}