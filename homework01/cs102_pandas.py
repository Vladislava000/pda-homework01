import pandas as pd
import re
from typing import Tuple

# Задача 1

def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    mask = data["факультет"].str.contains("систем управления и робототехники", case=False, na=False)
    fsuir = data[mask].copy()
    return len(fsuir), fsuir["группа"].nunique(), fsuir

# Задача 2

def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    """
    Проверяет наличие однофамильцев на ФСУиР, их количество, распределение по курсам
    и определяет группу с наибольшим числом однофамильцев.
    Возвращает:
    - логическое значение (наличие однофамильцев)
    - общее количество однофамильцев
    - серию с числом однофамильцев по курсам
    - группу с максимальным числом однофамильцев
    """
    mask = df["surname"].duplicated(keep=False)
    homonyms = df[mask]
    # на каждом курсе считаем однофамильцев среди студентов этого курса
    per_course = df.groupby("курс")["surname"].apply(lambda s: int(s.duplicated(keep=False).sum()))
    max_group = homonyms.groupby("группа").size().idxmax() if not homonyms.empty else ""
    return bool(mask.any()), int(mask.sum()), per_course, max_group

# Задача 3

def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    p = str(patronym).strip().lower()
    if p.endswith(("ович", "евич", "ич")):
        return "male"
    if p.endswith(("овна", "евна", "ична", "инична")):
        return "female"
    return "unknown"

def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
    - количество студентов без отчества
    - серию с распределением студентов по полу
    """
    no_patronym = df["patronim"].fillna("").str.strip() == ""
    with_patronym = df[~no_patronym]
    gender_counts = with_patronym["patronim"].map(gender_identification).value_counts()
    return int(no_patronym.sum()), gender_counts

# Задача 4

def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    counts = data["факультет"].value_counts()
    faculty_counts = counts.rename("количество").reset_index()
    return (
        faculty_counts,
        (counts.idxmax(), int(counts.max())),
        (counts.idxmin(), int(counts.min())),
    )

# Задача 5

def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    # число студентов на каждом факультете в рамках курса, затем среднее/медиана по курсам
    faculty_sizes = data.groupby(["курс", "факультет"]).size()
    grouped = faculty_sizes.groupby(level="курс")
    return grouped.mean(), grouped.median()

# Задача 6

def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, str, float]:
    """
    Определяет самое популярное имя, группу с наибольшим количеством студентов с этим именем,
    факультет, курс и долю таких студентов в общем числе.
    Возвращает результат в следующем порядке:
    1. самое частое имя
    2. группа
    3. факультет
    4. курс
    5. доля
    """
    name = data["name"].value_counts().idxmax()
    with_name = data[data["name"] == name]
    group = with_name["группа"].value_counts().idxmax()
    row = data[data["группа"] == group].iloc[0]
    ratio = round(len(with_name) / len(data), 2)
    return name, group, row["факультет"], row["курс"], ratio

# Задача 7

def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    unique_name = ~data["name"].duplicated(keep=False)
    starts_with_p = data["name"].str.startswith("П", na=False)
    return data.loc[unique_name & starts_with_p, ["фио", "факультет", "курс"]]

# Задача 8

def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    third = data[data["курс"] == "3-й"]
    faculty = third.groupby("факультет")["средний_балл"].mean().idxmax()

    top = third[third["факультет"] == faculty].copy()
    top["gender"] = top["patronim"].map(gender_identification)
    by_gender = top[top["gender"] != "unknown"].groupby("gender")["средний_балл"].mean()
    return faculty, by_gender.idxmax(), int(round(by_gender.max()))

# Задача 9

def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    s = data.sort_values("ису").reset_index(drop=True)
    # номер отличается от предыдущего ровно на 1 и курс тот же
    step_ok = ((s["ису"].diff() == 1) & (s["курс"] == s["курс"].shift())).astype(int)
    # 4 шага подряд = 5 студентов с последовательными номерами
    run = step_ok.rolling(4).sum() == 4
    if not run.any():
        return s.iloc[0:0][["ису", "фио", "факультет", "курс", "группа"]]
    end = run.idxmax()
    return s.loc[end - 4:end, ["ису", "фио", "факультет", "курс", "группа"]]

if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")

    parts = data["фио"].str.split()
    data["surname"] = parts.str[0]
    data["name"] = parts.str[1]
    data["patronim"] = parts.str[2:].str.join(" ")

    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")

    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = find_homonymous_students(fsuir)
    print(f"Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}")
    print(f"На каждом курсе: {homonyms_per_course}")

    # Задача 3
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print(f"Студентов без отчества: {students_without_patronym}")
    print("Распределение по полу:", gender_counts)

    # Задача 4
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print(f"Факультет с наибольшим числом студентов: {max_faculty}")
    print(f"Факультет с наименьшим числом студентов: {min_faculty}")

    # Задача 5
    mean_students, median_students = course_statistics(data)
    print("Среднее число студентов на курсах:", mean_students)
    print("Медианное число студентов на курсах:", median_students)

    # Задача 6
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print(f"Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}")
    print(f"Доля студентов с этим именем: {name_ratio}")

    # Задача 7
    result_7 = find_students_with_name_starting_P(data)
    print("Студенты с именем, начинающимся на П и встречающимся ровно один раз:")
    print(result_7)

    # Задача 8
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print(f"Факультет с высоким средним баллом 3-го курса: {fac}")
    print(f"Пол с наивысшим средним баллом: {best_gender}, Средний балл: {best_grade}")

    # Задача 9
    result_9 = find_consecutive_students(data)
    print("Первые 5 студентов с подряд идущими табельными номерами:")
    print(result_9)
