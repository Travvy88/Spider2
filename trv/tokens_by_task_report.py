import os
import re
import fnmatch
import subprocess
import csv
from typing import List, Tuple, Dict

from tqdm import tqdm

# Add imports from trv.context
from context import estimate_gemini_tokens

def run_context_py_and_parse_results(folder_path: str, exclude_patterns: List[str], exclude_table_patterns: List[str]) -> Tuple[int, int, int, Dict[str, int]]:
    """
    Запускает context.py для указанной папки и парсит результаты из context.txt.
    Возвращает: (total_tokens, business_context_tokens, db_tokens, file_type_tokens)
    """
    # Подготавливаем аргументы для context.py
    cmd = [
        "python", "trv/context.py",
        folder_path,
        "--output_file", "context.txt"
    ]

    # Добавляем шаблоны исключений
    for pattern in exclude_patterns:
        cmd.extend(["-e", pattern])
    for pattern in exclude_table_patterns:
        cmd.extend(["-et", pattern])

    try:
        # Запускаем context.py
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"Ошибка при запуске context.py для {folder_path}: {result.stderr}")
            return 0, 0, 0, {}

        # Парсим содержимое из stdout (отчет о токенах)
        return parse_context_output(result.stdout)

    except Exception as e:
        print(f"Произошла ошибка при обработке {folder_path}: {e}")
        return 0, 0, 0, {}

def parse_context_output(output: str) -> Tuple[int, int, int, Dict[str, int]]:
    """
    Парсит вывод context.py и возвращает статистику по токенам.
    Возвращает: (total_tokens, business_context_tokens, db_tokens, file_type_tokens)
    """
    total_tokens = 0
    business_context_tokens = 0
    db_tokens = 0
    file_type_tokens = {}

    # Ищем таблицу с токенами в выводе
    lines = output.split('\n')
    in_table = False

    for line in lines:
        line = line.strip()
        if line.startswith('| Категория |'):
            in_table = True
            continue
        elif line.startswith('---') and in_table:
            break
        elif in_table and line.startswith('|'):
            # Парсим строку таблицы
            parts = [part.strip() for part in line.split('|')[1:-1]]  # Убираем пустые части от | в начале и конце
            if len(parts) >= 3:
                category, file_type, tokens_str = parts[0], parts[1], parts[2]

                # Убираем запятые из числа токенов
                try:
                    tokens = int(tokens_str.replace(',', '').replace('**', ''))
                except ValueError:
                    continue

                if category == '**ИТОГО**':
                    total_tokens = tokens
                elif category == '**Весь бизнес-контекст**':
                    business_context_tokens = tokens
                elif category == 'Бизнес-контекст':
                    # Это тип файла
                    file_type_tokens[file_type] = tokens
                elif category == 'Базы данных DuckDB':
                    db_tokens = tokens

    return total_tokens, business_context_tokens, db_tokens, file_type_tokens

def main():
    examples_dir = "spider2-dbt/examples/"
    results = []
    file_type_results = {}

    print(f"Сканирование папки: {examples_dir}")

    # Define default exclude patterns if needed, otherwise empty
    common_exclude_patterns = []
    common_exclude_table_patterns = []

    # Собираем все уникальные типы файлов
    all_file_types = set()

    for entry_name in tqdm(sorted(os.listdir(examples_dir))):
        full_path = os.path.join(examples_dir, entry_name)
        if os.path.isdir(full_path):
            print(f"Обработка папки: {entry_name}")

            try:
                # Запускаем context.py и получаем результаты
                total_tokens, business_context_tokens, db_tokens, file_type_tokens = run_context_py_and_parse_results(
                    full_path, common_exclude_patterns, common_exclude_table_patterns
                )

                results.append((entry_name, total_tokens, business_context_tokens, db_tokens))
                file_type_results[entry_name] = file_type_tokens

                # Собираем все типы файлов
                all_file_types.update(file_type_tokens.keys())

            except Exception as e:
                print(f"Произошла непредвиденная ошибка для {entry_name}: {e}")
                results.append((entry_name, "ОШИБКА", "ОШИБКА", "ОШИБКА"))

    # Генерация первого CSV: папка - всего токенов - бизнес контекст - базы данных
    with open("trv/tokens_summary.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Папка", "Всего токенов", "Бизнес контекст", "Базы данных"])
        for folder, total_tokens, business_tokens, db_tokens in sorted(results, key=lambda x: x[0]):
            if total_tokens == "ОШИБКА":
                writer.writerow([folder, "ОШИБКА", "ОШИБКА", "ОШИБКА"])
            else:
                writer.writerow([folder, total_tokens, business_tokens, db_tokens])

    # Генерация второго CSV: папка - всего токенов - колонки на каждый тип файла
    with open("trv/tokens_by_filetype.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["Папка", "Всего токенов"] + sorted(all_file_types)
        writer.writerow(header)

        for folder, total_tokens, _, _ in sorted(results, key=lambda x: x[0]):
            if total_tokens == "ОШИБКА":
                row = [folder, "ОШИБКА"] + ["ОШИБКА"] * len(all_file_types)
            else:
                row = [folder, total_tokens]
                for file_type in sorted(all_file_types):
                    tokens = file_type_results[folder].get(file_type, 0)
                    row.append(tokens)
            writer.writerow(row)

    print("Отчеты сохранены в файлы: tokens_summary.csv и tokens_by_filetype.csv")

if __name__ == "__main__":
    main()

