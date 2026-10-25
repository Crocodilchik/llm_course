from pathlib import Path
import yaml

def test_compose_keeps_data_local_and_db_private():
    root=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((root/'compose.yaml').read_text())
    assert 'ports' not in config['services']['db']
    assert config['services']['app']['ports']==['127.0.0.1:8000:8000']
    assert all(v.split(':')[0].startswith('../') for service in config['services'].values() for v in service.get('volumes',[]))
    assert config['services']['db']['environment']['POSTGRES_PASSWORD_FILE']=='/run/secrets/db_password'
    dockerignore=(root/'.dockerignore').read_text()
    assert '.env' in dockerignore and '.venv' in dockerignore and 'private' in dockerignore
