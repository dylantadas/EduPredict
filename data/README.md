# Open University Learning Analytics Dataset (OULAD)

## Overview

The Open University Learning Analytics Dataset (OULAD) contains data about courses, students, and their interactions with Virtual Learning Environment (VLE) for seven selected courses (modules). The dataset contains information about 32,593 students, their assessment results, and logs of their interactions with the VLE represented by daily summaries of student clicks (10,655,280 entries). The dataset is anonymized and is suitable for a range of learning analytics and educational data mining studies.

## Dataset Components

The dataset consists of seven data files:

1. **courses.csv** - Information about the modules/courses
2. **assessments.csv** - Information about assessments in each module
3. **vle.csv** - Information about learning materials in the VLE
4. **studentInfo.csv** - Demographic information about the students
5. **studentRegistration.csv** - Information about the students' registration on the modules
6. **studentAssessment.csv** - Results of students' assessments
7. **studentVle.csv** - Students' interactions with the VLE

## Data Files and Variables

### courses.csv
Information about modules (courses).

| Variable | Description |
|----------|-------------|
| code_module | Code name of the module, which serves as the identifier. |
| code_presentation | Code name of the presentation (semester). Example: 2013B refers to the second semester of 2013. |
| length | Length of the module in days. |

### assessments.csv
Information about assessments in each module.

| Variable | Description |
|----------|-------------|
| code_module | Module identifier. |
| code_presentation | Presentation identifier. |
| id_assessment | Assessment identifier. |
| assessment_type | Type of assessment: Tutor Marked Assessment (TMA), Computer Marked Assessment (CMA), or Final Exam (Exam). |
| date | Information about the cut-off date of the assessment calculated as the number of days since the start of the module. |
| weight | Weight of the assessment in the overall score for the final result. |

### vle.csv
Information about learning materials/activities in the VLE.

| Variable | Description |
|----------|-------------|
| id_site | Identifier of the VLE material. |
| code_module | Module identifier. |
| code_presentation | Presentation identifier. |
| activity_type | The type of learning activity (resource, URL, quiz, etc.). |
| week_from | Week from which the material is planned to be used. |
| week_to | Week until which the material is planned to be used. |

### studentInfo.csv
Demographic information about the students and their final results.

| Variable | Description |
|----------|-------------|
| code_module | Module identifier. |
| code_presentation | Presentation identifier. |
| id_student | Student identifier. |
| gender | Student's gender. |
| region | Geographic region of the student. |
| highest_education | Highest student education level on entry. |
| imd_band | Index of multiple deprivation band of the student: 0-10%, 10-20%, ..., 90-100%, where 0-10% is the most deprived. |
| age_band | Student's age band: 0-35, 35-55, 55≤ |
| num_of_prev_attempts | Number of previous attempts at this module. |
| studied_credits | Total number of credits being studied (including the current module). |
| disability | Indicates if the student has declared a disability. |
| final_result | Student's final result in the module: Pass, Fail, Withdrawn, or Distinction. |

### studentRegistration.csv
Information about the time when the student registered for the module presentation.

| Variable | Description |
|----------|-------------|
| code_module | Module identifier. |
| code_presentation | Presentation identifier. |
| id_student | Student identifier. |
| date_registration | Date of student's registration on the module presentation, measured as the number of days from the start of the module-presentation. Negative values mean that the registration is before the official start of the module-presentation. |
| date_unregistration | Date of student unregistration from the module presentation, measured as the number of days from the start of the module-presentation. Students who completed the module have this field empty. |

### studentAssessment.csv
Results of students' assessments.

| Variable | Description |
|----------|-------------|
| id_assessment | Assessment identifier. |
| id_student | Student identifier. |
| date_submitted | Date of submission, measured as the number of days from the start of the module-presentation. |
| is_banked | Indicates whether the assessment result has been transferred from a previous presentation (True/False). |
| score | Student's score in this assessment. Range is from 0 to 100. The pass rate is typically 40. |

### studentVle.csv
Student interactions with the VLE and learning materials.

| Variable | Description |
|----------|-------------|
| code_module | Module identifier. |
| code_presentation | Presentation identifier. |
| id_student | Student identifier. |
| id_site | VLE material identifier. |
| date | Date of interaction, measured as the number of days from the start of the module-presentation. |
| sum_click | Number of interactions/clicks on this date with the specific VLE material. |

## Data Format and Relationships

The dataset tables are related through various foreign keys:
- The *code_module* and *code_presentation* fields link almost all tables
- The *id_student* field connects student-related tables
- The *id_assessment* field links assessment information with student performance
- The *id_site* field relates VLE materials to student interactions

## Value Mappings

### code_module
- AAA: Social sciences
- BBB: STEM (Science, Technology, Engineering, and Mathematics)
- CCC: STEM
- DDD: STEM
- EEE: STEM
- FFF: STEM
- GGG: Social sciences

### code_presentation
- 2013B: Second semester of 2013 (starts in February)
- 2013J: First semester of 2013 (starts in October)
- 2014B: Second semester of 2014
- 2014J: First semester of 2014

### final_result
- Distinction: Passed with distinction
- Pass: Passed
- Fail: Failed
- Withdrawn: Student withdrew from the module

### assessment_type
- TMA: Tutor Marked Assessment
- CMA: Computer Marked Assessment
- Exam: Final Exam

## Usage Recommendations

1. **Data Preprocessing**:
   - Handle missing values appropriately
   - Convert date fields to appropriate formats
   - Consider normalizing numeric fields

2. **Potential Research Areas**:
   - Predicting student performance
   - Analyzing engagement patterns
   - Identifying at-risk students
   - Understanding the impact of demographic factors on education outcomes

3. **Ethical Considerations**:
   - Though anonymized, respect student privacy
   - Be cautious about drawing deterministic conclusions about demographic factors

## Citation

If you use this dataset, please cite:
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. Scientific Data, 4(1), 1-8.

## License

The dataset is available under the CC-BY 4.0 license.

## Acknowledgments

This dataset was created and shared by the Knowledge Media Institute (KMi) and Institute of Educational Technology at The Open University, UK.
