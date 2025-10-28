import os
# Remove subprocess and re imports
# import subprocess
# import re

# Add imports from trv.context
from trv.context import collect_and_separate_contents, estimate_gemini_tokens, generate_file_tree

def main():
    examples_dir = "/home/travvy/Spider2/spider2-dbt/examples/"
    # Remove context_script_path since we'll be importing directly
    # context_script_path = "/home/travvy/Spider2/trv/context.py"
    results = []

    print(f"Сканирование папки: {examples_dir}")

    # Define default exclude patterns if needed, otherwise empty
    # For now, let's assume no specific global excludes for these examples
    # The functions in context.py already handle binary files and dot files.
    common_exclude_patterns = []
    common_exclude_table_patterns = []


    for entry_name in os.listdir(examples_dir):
        full_path = os.path.join(examples_dir, entry_name)
        if os.path.isdir(full_path):
            print(f"Обработка папки: {entry_name}")
            # Remove output_file related code
            # output_file = f"temp_context_{entry_name}.txt"

            try:
                # 1. Генерация дерева файлов (as done in context.py's main)
                file_tree_str = "\n".join(generate_file_tree(full_path, common_exclude_patterns))
                tree_header = "[FILE TREE]"
                tree_footer = "[END FILE TREE]"
                full_tree_block = f"{tree_header}\n{file_tree_str}\n{tree_footer}\n\n"

                # 2. Сбор содержимого, разделяя текст и БД
                text_contents, db_contents = collect_and_separate_contents(
                    full_path, common_exclude_patterns, common_exclude_table_patterns
                )
                full_text_block = "\n".join(text_contents)
                full_db_block = "\n".join(db_contents)

                # 3. Подсчет токенов
                tokens_tree = estimate_gemini_tokens(full_tree_block)
                tokens_text = estimate_gemini_tokens(full_text_block)
                tokens_db = estimate_gemini_tokens(full_db_block)
                total_tokens = tokens_tree + tokens_text + tokens_db

                results.append((entry_name, total_tokens))

            except Exception as e:
                print(f"Произошла непредвиденная ошибка для {entry_name}: {e}")
                results.append((entry_name, "ОШИБКА"))

    # Генерация таблицы Markdown
    print("\n### Отчет по токенам")
    print("| Папка | Количество токенов |")
    print("|-------|--------------------|")
    for folder, tokens in sorted(results, key=lambda x: x[0]):
        print(f"| {folder} | {tokens:,} |")

if __name__ == "__main__":
    main()

