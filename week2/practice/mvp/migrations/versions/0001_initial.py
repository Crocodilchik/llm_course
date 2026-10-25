"""Initial MVP tables. ADK manages its own session tables."""
from alembic import op
from sqlalchemy import Column,String,Text,DateTime,Integer,Boolean,JSON,ForeignKey,UniqueConstraint
from pgvector.sqlalchemy import Vector
revision='0001'
down_revision=None
branch_labels=None
depends_on=None

def upgrade():
    op.create_table('mvp_users',Column('id',String(32),primary_key=True),Column('username',String(32),nullable=False,unique=True),Column('password_hash',Text,nullable=False),Column('created_at',DateTime(timezone=True),nullable=False))
    op.create_table('mvp_auth_sessions',Column('digest',String(64),primary_key=True),Column('user_id',String(32),ForeignKey('mvp_users.id'),nullable=False,index=True),Column('csrf_digest',String(64),nullable=False),Column('expires_at',DateTime(timezone=True),nullable=False),Column('revoked',Boolean,nullable=False))
    op.create_table('mvp_chats',Column('id',String(32),primary_key=True),Column('user_id',String(32),ForeignKey('mvp_users.id'),nullable=False,index=True),Column('title',String(100),nullable=False),Column('created_at',DateTime(timezone=True),nullable=False),Column('last_results',JSON,nullable=False),Column('filters',JSON,nullable=False))
    op.create_table('mvp_runs',Column('id',String(32),primary_key=True),Column('chat_id',String(32),ForeignKey('mvp_chats.id'),nullable=False,index=True),Column('user_id',String(32),ForeignKey('mvp_users.id'),nullable=False,index=True),Column('request_key',String(64),nullable=False),Column('prompt',Text,nullable=False),Column('status',String(20),nullable=False),Column('final_answer',JSON),Column('error_code',String(50)),Column('sequence',Integer,nullable=False),Column('model_calls',Integer,nullable=False),Column('input_tokens',Integer),Column('output_tokens',Integer),Column('created_at',DateTime(timezone=True),nullable=False),UniqueConstraint('chat_id','request_key'))
    op.create_table('mvp_run_events',Column('run_id',String(32),ForeignKey('mvp_runs.id'),primary_key=True),Column('sequence',Integer,primary_key=True),Column('kind',String(40),nullable=False),Column('payload',JSON,nullable=False),Column('created_at',DateTime(timezone=True),nullable=False))
    op.create_table('mvp_search_documents',Column('poi_id',Integer,primary_key=True),Column('version',String(64),primary_key=True),Column('document',Text,nullable=False),Column('embedding',Vector(384).with_variant(JSON(),'sqlite'),nullable=False),Column('attributes',JSON,nullable=False))
    op.create_table('mvp_metadata',Column('key',String(80),primary_key=True),Column('value',JSON,nullable=False))

def downgrade():
    raise RuntimeError('Destructive downgrade requires a separately approved backup and migration.')
