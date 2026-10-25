from alembic import context
from app.models import Base

def migrate(connection):
    context.configure(connection=connection,target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()

connection=context.config.attributes.get('connection')
if connection is None:
    raise RuntimeError('Run migrations through application startup; configuration is never printed.')
migrate(connection)
