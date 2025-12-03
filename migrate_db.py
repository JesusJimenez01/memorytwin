"""Script de migración para añadir columnas faltantes a la BD."""
from sqlalchemy import create_engine, text
from memorytwin.config import get_sqlite_path

engine = create_engine(f'sqlite:///{get_sqlite_path()}')
with engine.connect() as conn:
    # Verificar columnas existentes
    result = conn.execute(text('PRAGMA table_info(episodes)'))
    columns = [row[1] for row in result.fetchall()]
    print('Columnas existentes:', columns)
    
    # Añadir columnas faltantes si no existen
    new_columns = [
        ('is_antipattern', 'BOOLEAN DEFAULT 0'),
        ('is_critical', 'BOOLEAN DEFAULT 0'),
        ('superseded_by', 'VARCHAR(36)'),
        ('deprecation_reason', 'TEXT')
    ]
    
    for col_name, col_type in new_columns:
        if col_name not in columns:
            try:
                conn.execute(text(f'ALTER TABLE episodes ADD COLUMN {col_name} {col_type}'))
                conn.commit()
                print(f'✅ Añadida columna: {col_name}')
            except Exception as e:
                print(f'⚠️ Error añadiendo {col_name}: {e}')
        else:
            print(f'✓ Columna ya existe: {col_name}')

print('\n✅ Migración completada')
