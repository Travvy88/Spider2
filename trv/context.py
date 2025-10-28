import os
import argparse
import fnmatch
import duckdb
from transformers import AutoTokenizer
import traceback
from typing import List, Tuple

# --- Вспомогательные функции ---

# Инициализация токенизатора Gemma
_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


def estimate_gemini_tokens(text: str) -> int:
    """
    Оценивает количество токенов для Gemini, используя токенизатор Gemma, если он доступен,
    иначе использует простое правило: 1 токен ~ 4 символа.
    """
    if not text: return 0
    if _tokenizer:
        # Токенизатор Gemma разбивает текст на токены и возвращает их количество
        return len(_tokenizer.encode(text))
    else:
        # Fallback к простому правилу, если токенизатор Gemma недоступен
        return len(text) // 4

def is_binary(filepath: str) -> bool:
    """
    Проверяет, является ли файл бинарным, ища нулевой байт в начале файла.
    """
    try:
        with open(filepath, 'rb') as f:
            return b'\x00' in f.read(1024)
    except IOError:
        return True

def is_excluded(path: str, exclude_patterns: List[str]) -> bool:
    """
    Проверяет, соответствует ли путь (файл или папка) одному из шаблонов исключения.
    """
    path_basename = os.path.basename(path)
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path_basename, pattern):
            return True
    return False

# --- Функции для работы с файловой системой ---

def generate_file_tree(start_path: str, exclude_patterns: List[str], prefix: str = "") -> List[str]:
    """
    Рекурсивно генерирует строки с деревом файлов, учитывая исключения.
    """
    tree_lines = []
    try:
        entries = sorted([e for e in os.listdir(start_path) if not is_excluded(os.path.join(start_path, e), exclude_patterns)])
    except FileNotFoundError:
        return []

    for i, entry in enumerate(entries):
        connector = "├── " if i < len(entries) - 1 else "└── "
        tree_lines.append(f"{prefix}{connector}{entry}")
        
        full_path = os.path.join(start_path, entry)
        if os.path.isdir(full_path):
            extension = "│   " if i < len(entries) - 1 else "    "
            tree_lines.extend(generate_file_tree(full_path, exclude_patterns, prefix + extension))
            
    return tree_lines

# --- Функции для работы с DuckDB ---

def get_db_tables(cursor) -> List[str]:
    cursor.execute("SHOW TABLES;")
    return [table[0] for table in cursor.fetchall()]

def get_db_columns_info(cursor, table_name: str) -> Tuple[List[str], str]:
    cursor.execute(f'PRAGMA table_info("{table_name}");')
    columns = cursor.fetchall()
    names = [col[1] for col in columns]
    info_str = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
    return names, info_str

def get_db_primary_key_info(cursor, table_name: str) -> Tuple[str, int]:
    cursor.execute(f'PRAGMA table_info("{table_name}");')
    columns_info = cursor.fetchall()
    for col in columns_info:
        if col[5]:
            return col[1], col[0]
    return "ID", 0

def format_db_row(row: tuple, column_names: List[str], pk_name: str, pk_index: int) -> str:
    if not row or not column_names: return ""
    pk_value = row[pk_index] if row and len(row) > pk_index else "N/A"
    pk_tag = pk_name.upper()
    start_part = f"{pk_tag}: {pk_value}"
    end_part = f"END {pk_tag}: {pk_value}"
    content_parts = [f"{col_name.upper()}: {row[i] if row[i] is not None else ''}" for i, col_name in enumerate(column_names) if i != pk_index]
    return " | ".join([start_part] + content_parts + [end_part])

# --- Основная логика сбора контента ---

def process_database_file(db_path: str, exclude_table_patterns: List[str]) -> str:
    """
    Обрабатывает один файл базы данных DuckDB и возвращает его содержимое в виде строки.
    """
    relative_path = os.path.relpath(db_path)
    db_content = [f"[DATABASE {relative_path}]"]
    try:
        conn = duckdb.connect(db_path, read_only=True)
        cursor = conn.cursor()
        tables = get_db_tables(cursor)
        if not tables:
            db_content.append(f"INFO: В базе данных таблицы не найдены.")
        
        for table_name in tables:
            if is_excluded(table_name, exclude_table_patterns): continue
            db_content.append(f"[TABLE {table_name}]")
            column_names, columns_str = get_db_columns_info(cursor, table_name)
            db_content.append(f"COLUMNS: {columns_str}")
            pk_name, pk_index = get_db_primary_key_info(cursor, table_name)
            
            cursor.execute(f'SELECT * FROM "{table_name}";')
            rows = cursor.fetchall()
            for row in rows:
                db_content.append(format_db_row(row, column_names, pk_name, pk_index))
            db_content.append(f"[END TABLE {table_name}]")
        conn.close()
    except duckdb.Error as e:
        db_content.append(f"ERROR: Не удалось прочитать базу данных. Ошибка: {e}")
    
    db_content.append(f"[END DATABASE {relative_path}]\n")
    return "\n".join(db_content)

def collect_and_separate_contents(start_path: str, exclude_patterns: List[str], exclude_table_patterns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Рекурсивно обходит директорию, собирая содержимое текстовых файлов
    и баз данных в два отдельных списка.
    """
    text_contents = []
    db_contents = []
    
    for root, dirs, files in os.walk(start_path, topdown=True):
        dirs[:] = [d for d in dirs if not is_excluded(os.path.join(root, d), exclude_patterns)]
        
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            if is_excluded(file_path, exclude_patterns): continue
            
            relative_path = os.path.relpath(file_path, start_path)

            if filename.endswith((".db", ".duckdb")):
                db_contents.append(process_database_file(file_path, exclude_table_patterns))
                continue

            if is_binary(file_path): continue

            content_block = [f"[{relative_path}]"]
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content_block.append(f.read())
            except Exception as e:
                content_block.append(f"Не удалось прочитать файл: {e}")
            
            content_block.append(f"[END {relative_path}]\n")
            text_contents.append("\n".join(content_block))
            
    return text_contents, db_contents

# --- Точка входа ---

def main():
    parser = argparse.ArgumentParser(
        description="Собирает дерево файлов, текстовое содержимое и данные из баз DuckDB в один контекстный файл.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # ... (аргументы без изменений)
    parser.add_argument('root_dir', help="Корневая директория для сканирования.")
    parser.add_argument('--output_file', '-o', default='context.txt', help="Имя итогового файла (по умолчанию: context.txt).")
    parser.add_argument('-e', '--exclude', action='append', default=[], help="Шаблон для исключения файлов или папок.")
    parser.add_argument('-et', '--exclude-table', action='append', default=[], dest='exclude_table_patterns', help="Шаблон для исключения таблиц из баз данных.")
    
    args = parser.parse_args()
    
    try:
        script_name = os.path.basename(__file__)
        args.exclude.append(script_name)
    except NameError: pass 
    args.exclude.append(args.output_file)
    
    print(f"Сканирование директории: {os.path.abspath(args.root_dir)}")
    print(f"Исключаемые шаблоны файлов/папок: {args.exclude}")
    print(f"Исключаемые шаблоны таблиц БД: {args.exclude_table_patterns}")
    
    # 1. Генерация дерева файлов
    print("1. Генерация дерева файлов...")
    tree_header = "[FILE TREE]"
    tree_footer = "[END FILE TREE]"
    file_tree_str = "\n".join(generate_file_tree(args.root_dir, args.exclude))
    
    # 2. Сбор содержимого, разделяя текст и БД
    print("2. Сбор содержимого файлов и данных из БД...")
    text_contents, db_contents = collect_and_separate_contents(
        args.root_dir, args.exclude, args.exclude_table_patterns
    )
    
    # <-- НОВОЕ: Подсчет токенов -->
    print("3. Подсчет токенов...")
    
    # Собираем все части в строки для подсчета и записи
    full_tree_block = f"{tree_header}\n{file_tree_str}\n{tree_footer}\n\n"
    full_text_block = "\n".join(text_contents)
    full_db_block = "\n".join(db_contents)

    # Считаем токены для каждой части
    tokens_tree = estimate_gemini_tokens(full_tree_block)
    tokens_text = estimate_gemini_tokens(full_text_block)
    tokens_db = estimate_gemini_tokens(full_db_block)
    total_tokens = tokens_tree + tokens_text + tokens_db

    # Выводим статистику
    print("\n----------------------------------------")
    print("--- Анализ токенов (Gemini, приблиз.) ---")
    print(f"Дерево файлов:      {tokens_tree:>8,} токенов")
    print(f"Текстовые файлы:    {tokens_text:>8,} токенов")
    print(f"Базы данных:        {tokens_db:>8,} токенов")
    print("----------------------------------------")
    print(f"ИТОГО:              {total_tokens:>8,} токенов")
    print("----------------------------------------\n")

    # 4. Запись результата в файл
    print(f"4. Запись результата в файл '{args.output_file}'...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(full_tree_block)
            if full_text_block:
                f.write(full_text_block)
            if full_db_block:
                f.write(full_db_block)
                
        print("Готово! Файл успешно создан.")
    except IOError as e:
        print(f"Ошибка при записи в файл: {e}")

if __name__ == "__main__":
    main()