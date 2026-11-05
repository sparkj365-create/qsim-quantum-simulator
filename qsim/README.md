# qsim — شبیه‌ساز مدار کوانتومی با پایتون

یک شبیه‌ساز state-vector که فقط با NumPy نوشته شده و چند الگوریتم کلاسیک کوانتومی روی آن پیاده شده است.

## ساختار

```
quantum_project/
├── qsim/
│   ├── circuit.py      # QuantumCircuit: گیت‌ها، اندازه‌گیری، نمونه‌برداری
│   └── algorithms.py   # Bell, GHZ, Deutsch-Jozsa, Grover, QFT, Teleportation
├── tests/test_qsim.py  # ۱۲ تست
└── main.py             # اجرای دموی همه الگوریتم‌ها
```

## اجرا

```bash
pip install numpy pytest
python main.py
python -m pytest -q tests
```

## نمونه استفاده

```python
from qsim import QuantumCircuit

qc = QuantumCircuit(2, seed=1)
qc.h(0).cx(0, 1)
print(qc.sample(1000))   # فقط '00' و '11'
```

## قرارداد کیوبیت‌ها

کیوبیت ۰ پرارزش‌ترین بیت (MSB) است: حالت `|q0 q1 ... q(n-1)>` در اندیس `q0*2^(n-1)+...+q(n-1)` قرار دارد.

## ایده‌های توسعه

- نویز و کانال‌های دپولاریزاسیون (ماتریس چگالی)
- الگوریتم شُر (Shor) برای تجزیه اعداد کوچک
- VQE / QAOA و اتصال به ابزارهای یادگیری ماشین (مثلاً بهینه‌سازی پارامترها با gradient descent)
