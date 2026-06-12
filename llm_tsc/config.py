'''
@Author: WANG Maonan
@Date: 2026-06-11 17:05:37
@Description: Configuration Management using Environment Variables
@LastEditTime: 2026-06-12 23:05:29
'''
import os
import yaml

from pathlib import Path
from typing import Dict, Any, Optional
from llm_tsc.logger import logger

class Config:
    """
    Configuration manager supporting both environment variables and YAML files
    Prioritizes environment variables over YAML for security
    
    Supports:
    - OpenAI API settings
    - Simulation parameters
    - Agent settings
    """
    
    # Default config values
    DEFAULTS = {
        # ==================
        # OpenAI Settings
        # ==================
        'OPENAI_API_KEY': '',
        'OPENAI_API_MODEL': 'gpt-4',
        'OPENAI_API_BASE': 'https://api.openai.com/v1',
        'OPENAI_PROXY': None,
        'OPENAI_TEMPERATURE': 0.0,
        
        # ==================
        # Agent Settings
        # ==================
        'MAX_ITERATIONS': 8,
        'LLM_VERBOSE': True,
        
        # ==================
        # Simulation Parameters
        # ==================
        'TOTAL_SIMULATION_TIME': 500,  # Total seconds
    }
    
    def __init__(self, yaml_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            yaml_path: Path to config.yaml (optional)
                      Falls back to YAML_CONFIG_PATH env var or default location
        """
        self.config: Dict[str, Any] = {}
        self.yaml_path = None
        
        # Try to load from YAML first
        if yaml_path is None:
            yaml_path = os.environ.get(
                'YAML_CONFIG_PATH',
                str(Path(__file__).parent.parent / 'config.yaml')
            )
        
        if yaml_path and os.path.exists(yaml_path):
            self._load_yaml(yaml_path)
            self.yaml_path = yaml_path
        
        # Override with environment variables
        self._load_env()
        
        # Fill in defaults for missing values
        for key, default_val in self.DEFAULTS.items():
            if key not in self.config:
                self.config[key] = default_val
    
    def _load_yaml(self, yaml_path: str) -> None:
        """Load configuration from YAML file"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
                self.config.update(yaml_config)
                logger.info(f"✓ Loaded config from {yaml_path}")
        except Exception as e:
            logger.warning(f"Failed to load YAML config from {yaml_path}: {e}")
    
    def _load_env(self) -> None:
        """Load configuration from environment variables"""
        env_mappings = {
            # OpenAI settings
            'OPENAI_API_KEY': 'OPENAI_API_KEY',
            'OPENAI_API_MODEL': 'OPENAI_API_MODEL',
            'OPENAI_API_BASE': 'OPENAI_API_BASE',
            'OPENAI_PROXY': 'OPENAI_PROXY',
            'OPENAI_TEMPERATURE': ('OPENAI_TEMPERATURE', float),
            
            # Agent settings
            'MAX_ITERATIONS': ('MAX_ITERATIONS', int),
            'LLM_VERBOSE': ('LLM_VERBOSE', lambda x: x.lower() in ('true', '1', 'yes')),
            
            # Simulation parameters
            'TOTAL_SIMULATION_TIME': ('TOTAL_SIMULATION_TIME', int),
        }
        
        for config_key, env_spec in env_mappings.items():
            if isinstance(env_spec, tuple):
                env_key, type_converter = env_spec
            else:
                env_key = env_spec
                type_converter = None
            
            if env_key in os.environ:
                value = os.environ[env_key]
                if type_converter:
                    try:
                        value = type_converter(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Failed to convert {env_key}={value}")
                        continue
                self.config[config_key] = value
        
        # Log loaded env variables (excluding sensitive ones)
        if 'OPENAI_API_KEY' in os.environ:
            logger.info("✓ Loaded OPENAI_API_KEY from environment")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Dict-like access"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator"""
        return key in self.config
    
    def __repr__(self) -> str:
        # Don't print sensitive keys
        safe_config = {k: v for k, v in self.config.items() 
                      if k != 'OPENAI_API_KEY'}
        return f"Config({safe_config})"
    
    def validate(self) -> bool:
        """Validate required configuration"""
        required_keys = ['OPENAI_API_KEY']
        
        for key in required_keys:
            if not self.config.get(key):
                logger.error(f"Missing required config: {key}")
                logger.error("Set via environment variable or config.yaml")
                return False
        
        return True
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export config as dictionary"""
        config_dict = dict(self.config)
        if not include_secrets and 'OPENAI_API_KEY' in config_dict:
            config_dict['OPENAI_API_KEY'] = '***REDACTED***'
        return config_dict


def load_config(yaml_path: Optional[str] = None) -> Config:
    """
    Convenience function to load configuration
    
    Usage:
        config = load_config()
        api_key = config['OPENAI_API_KEY']
    """
    return Config(yaml_path)
