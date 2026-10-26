import pandas as pd
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90],
    "Previous_Marks": [40, 45, 50, 55, 60, 65, 70, 80],
    "Result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass"]
}
df = pd.DataFrame(data)
print(df)
X = df[["Hours", "Attendance", "Previous_Marks"]]
y = df["Result"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
new_student = pd.DataFrame(
    [[7, 85, 75]],
    columns=["Hours", "Attendance", "Previous_Marks"]
)
prediction = model.predict(new_student)
print(prediction)
test_prediction = model.predict(X_test)

print(test_prediction)
print(y_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, test_prediction)

print("Accuracy:", accuracy)
hours = int(input("Enter study hours: "))
attendance = int(input("Enter attendance: "))
previous_marks = int(input("Enter previous marks: "))
new_student = pd.DataFrame(
    [[hours, attendance, previous_marks]],
    columns=["Hours", "Attendance", "Previous_Marks"]
)
prediction = model.predict(new_student)

print("Predicted Result:", prediction[0])
if prediction[0] == "Pass":
    print("Student is predicted to PASS.")
else:
    print("Student is predicted to FAIL.")
    
