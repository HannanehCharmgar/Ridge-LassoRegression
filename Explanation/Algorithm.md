# رگرسیون Ridge و Lasso

---


---

## 1. رگرسیون خطی چیست؟

رگرسیون خطی یک الگوریتم یادگیری نظارت‌شده است که رابطه بین ویژگی‌ها و متغیر هدف را مدل می‌کند.

$\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon
\]$

هدف، کمینه‌کردن خطای مجموع مربعات (MSE) است.

---

## 2. مشکل هم‌خطی و بیش‌برازش

### هم‌خطی (Multicollinearity)
وقتی چند ویژگی به‌شدت همبسته باشند:
- ضرایب ناپایدار می‌شوند
- تفسیر مدل سخت می‌شود

### بیش‌برازش (Overfitting)
- عملکرد عالی روی داده‌ی آموزش
- عملکرد ضعیف روی داده‌ی جدید
- معمولاً به دلیل پیچیدگی زیاد مدل

---

## 3. رگرسیون Ridge چیست؟ (Ridge Regression)

Ridge Regression نسخه‌ی منظم‌شده‌ی رگرسیون خطی با جریمه L2 است.

### ایده اصلی
ضرایب بزرگ باعث حساسیت زیاد مدل می‌شوند.  
Ridge با کوچک‌کردن ضرایب، مدل را پایدارتر می‌کند.

### تابع هزینه Ridge
$\[
J(\beta) = \sum (y_i - \hat{y}_i)^2 + \alpha \sum \beta_j^2
\]$

### ویژگی‌ها
- ضرایب صفر نمی‌شوند
- همه ویژگی‌ها حفظ می‌شوند
- مناسب برای داده‌های با هم‌خطی بالا

---

## 4. رگرسیون Lasso چیست؟ (Lasso Regression)

Lasso Regression از جریمه L1 استفاده می‌کند.

### ایده اصلی
- استفاده از قدر مطلق ضرایب
- برخی ضرایب دقیقاً صفر می‌شوند
- انجام انتخاب ویژگی خودکار

### تابع هزینه Lasso
$\[
J(\beta) = \sum (y_i - \hat{y}_i)^2 + \alpha \sum |\beta_j|
\]$

### ویژگی‌ها
- مدل ساده و قابل تفسیر
- حذف ویژگی‌های کم‌اهمیت
- در هم‌خطی شدید ممکن است فقط یک ویژگی باقی بماند

---

## 5. مقایسه Ridge و Lasso

| ویژگی | Ridge (L2) | Lasso (L1) |
|------|------------|------------|
| نوع جریمه | مربع ضرایب | قدر مطلق ضرایب |
| صفر شدن ضرایب | ❌ | ✅ |
| انتخاب ویژگی | ❌ | ✅ |
| هم‌خطی | پایدار | ناپایدار |
| تفسیرپذیری | متوسط | بالا |

### مقایسه هندسی

![Ridge vs Lasso](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Lasso-and-ridge.jpg)

![Regularization Geometry](https://i.sstatic.net/jdxus.jpg)

---

## 6. پارامتر α (یا λ)

- α کوچک → نزدیک به رگرسیون خطی
- α بزرگ → مدل ساده‌تر
- انتخاب α معمولاً با Cross-Validation

---

## 7. مثال‌های مفهومی

### مثال Ridge
- پیش‌بینی قیمت خانه
- همه ویژگی‌ها حفظ می‌شوند
- ضرایب کوچک‌تر و پایدارتر

### مثال Lasso
- داده با ویژگی‌های زیاد
- بسیاری از ضرایب صفر می‌شوند
- فقط ویژگی‌های مهم باقی می‌مانند

---

## 8. مزایا و معایب

### مزایا
- کاهش بیش‌برازش
- تعمیم بهتر مدل
- Lasso: انتخاب ویژگی
- Ridge: مناسب هم‌خطی بالا

### معایب
- Ridge: مدل همچنان پیچیده
- Lasso: ناپایدار در هم‌خطی شدید

---

## 9. پیاده‌سازی ساده در پایتون

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("MSE Ridge:", mean_squared_error(y_test, ridge.predict(X_test)))

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print("MSE Lasso:", mean_squared_error(y_test, lasso.predict(X_test)))

