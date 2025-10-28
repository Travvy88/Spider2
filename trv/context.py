import os
import argparse
import fnmatch
import duckdb
from transformers import GemmaTokenizerFast
import traceback
from typing import List, Tuple, Dict
from tqdm import tqdm

# --- Вспомогательные функции ---

# Инициализация токенизатора Gemma
_tokenizer = GemmaTokenizerFast.from_pretrained("google/gemma-3-27b-it")


def estimate_gemini_tokens(text: str, batch_size: int = 500_000) -> int:
    """
    Оценивает количество токенов для Gemini, используя токенизатор Gemma, если он доступен,
    иначе использует простое правило: 1 токен ~ 4 символа.
    Обрабатывает текст по батчам для эффективности.
    """
    if not text: return 0

    if _tokenizer:
        # Разбиваем текст на батчи и подсчитываем токены постепенно
        total_tokens = 0
        # Разбиваем на батчи по batch_size символов
        batches = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]

        with tqdm(total=len(batches), desc="Подсчет токенов", unit="батч", disable=len(batches) < 2) as pbar:
            for batch in batches:
                if batch:  # Проверяем, что батч не пустой
                    total_tokens += len(_tokenizer.encode(batch))
                pbar.update(1)

        return total_tokens
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

def format_value(value):
    """
    Форматирует значение для читаемого вывода, включая nested структуры.
    """
    if value is None:
        return ""

    # Обрабатываем типы DuckDB
    if hasattr(value, '__class__'):
        class_name = value.__class__.__name__

        # STRUCT - вложенный объект
        if class_name == 'Struct':
            try:
                # Пытаемся преобразовать в читаемый формат
                items = []
                for key, val in value.items():
                    formatted_val = format_value(val)
                    items.append(f"{key}: {formatted_val}")
                return "{" + ", ".join(items) + "}"
            except:
                return str(value)

        # LIST/MAP - массивы и словари
        elif class_name in ['List', 'Map']:
            try:
                if hasattr(value, 'values'):
                    # Map
                    items = []
                    for k, v in value.items():
                        formatted_val = format_value(v)
                        items.append(f"{k}=>{formatted_val}")
                    return "{" + ", ".join(items) + "}"
                else:
                    # List
                    items = [format_value(item) for item in value]
                    return "[" + ", ".join(items) + "]"
            except:
                return str(value)

        # Другие типы - пробуем строку
        else:
            return str(value)
    else:
        # Простые типы
        return str(value)

def format_db_row(row: tuple, column_names: List[str], pk_name: str, pk_index: int) -> str:
    if not row or not column_names: return ""
    pk_value = format_value(row[pk_index] if row and len(row) > pk_index else "N/A")
    pk_tag = pk_name.upper()
    start_part = f"{pk_tag}: {pk_value}"
    end_part = f"END {pk_tag}: {pk_value}"
    content_parts = [f"{col_name.upper()}: {format_value(row[i]) if row[i] is not None else ''}" for i, col_name in enumerate(column_names) if i != pk_index]
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

def collect_and_separate_contents(start_path: str, exclude_patterns: List[str], exclude_table_patterns: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Рекурсивно обходит директорию, собирая содержимое текстовых файлов
    и баз данных. Бизнес-контекст группируются по типам (расширениям).
    """
    text_contents_by_type = {}
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

            # Определяем тип файла по расширению
            _, ext = os.path.splitext(filename)
            file_type = ext[1:] if ext else "no_extension"  # Убираем точку из расширения

            content_block = [f"[{relative_path}]"]
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content_block.append(f.read())
            except Exception as e:
                content_block.append(f"Не удалось прочитать файл: {e}")

            content_block.append(f"[END {relative_path}]\n")
            content_str = "\n".join(content_block)

            # Добавляем в соответствующий тип
            if file_type not in text_contents_by_type:
                text_contents_by_type[file_type] = []
            text_contents_by_type[file_type].append(content_str)

    return text_contents_by_type, db_contents

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
    text_contents_by_type, db_contents = collect_and_separate_contents(
        args.root_dir, args.exclude, args.exclude_table_patterns
    )

    # <-- НОВОЕ: Подсчет токенов -->
    print("3. Подсчет токенов...")

    # Собираем все части в строки для подсчета и записи
    full_tree_block = f"{tree_header}\n{file_tree_str}\n{tree_footer}\n\n"
    full_text_block = ""
    for file_type, contents in text_contents_by_type.items():
        full_text_block += "\n".join(contents)
    full_db_block = "\n".join(db_contents)

    # Считаем токены для дерева файлов
    print("Подсчет токенов для дерева файлов...")
    tokens_tree = estimate_gemini_tokens(full_tree_block)

    # Считаем токены по типам файлов
    tokens_by_type = {}
    total_text_tokens = 0
    for file_type, contents in text_contents_by_type.items():
        type_block = "\n".join(contents)
        print(f"Подсчет токенов для файлов типа '{file_type}'...")
        tokens = estimate_gemini_tokens(type_block)
        tokens_by_type[file_type] = tokens
        total_text_tokens += tokens

    # Считаем токены для баз данных
    print("Подсчет токенов для баз данных...")
    tokens_db = estimate_gemini_tokens(full_db_block)

    total_tokens = tokens_tree + total_text_tokens + tokens_db

    # Создаем детальный отчет о токенах в формате Markdown таблицы
    tokens_report = f"""
# Анализ токенов (Gemini, приблиз.)

| Категория | Тип файла | Количество токенов |
|-----------|-----------|-------------------|
"""

    # Собираем все записи для сортировки
    token_entries = []

    # Добавляем дерево файлов
    token_entries.append(("Дерево файлов", "", tokens_tree))

    # Добавляем типы файлов, отсортированные по количеству токенов (по убыванию)
    for file_type, tokens in sorted(tokens_by_type.items(), key=lambda x: x[1], reverse=True):
        type_name = f".{file_type}" if file_type != "no_extension" else "(без расширения)"
        token_entries.append(("Бизнес-контекст", type_name, tokens))

    # Добавляем итог по текстовым файлам
    if tokens_by_type:
        token_entries.append(("**Весь бизнес-контекст**", "", total_text_tokens))

    # Добавляем базы данных
    token_entries.append(("Базы данных DuckDB", "", tokens_db))

    # Добавляем итоговую строку
    token_entries.append(("**ИТОГО**", "", total_tokens))

    # Формируем таблицу
    for category, file_type, tokens in token_entries:
        if file_type:
            tokens_report += f"| {category} | {file_type} | {tokens:,} |\n"
        else:
            tokens_report += f"| {category} | | {tokens:,} |\n"

    tokens_report += "\n---\n"

    # Выводим отчет о токенах в консоль
    print(tokens_report)

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