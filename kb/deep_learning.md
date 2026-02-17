# Deep Learning — конспект

## 1. Что такое Deep Learning

**Deep Learning (глубокое обучение)** — это раздел машинного обучения, где модели используют **многослойные нейронные сети** для извлечения признаков и предсказаний.

Главная идея: модель сама учится представлениям (features) из данных, без ручного feature engineering.

---

## 2. Основные компоненты нейросети

### 2.1 Нейрон

* Формула:

```
y = activation(Wx + b)
```

* W — веса, b — bias, activation — функция активации

### 2.2 Слои

* Dense / Fully Connected
* Convolutional (CNN)
* Recurrent (RNN, LSTM, GRU)
* Transformer / Attention

### 2.3 Activation Functions

* ReLU: max(0, x) — простая и эффективная
* Sigmoid: 1 / (1 + exp(-x)) — для вероятностей
* Tanh: [-1,1] — центровка
* Softmax — для многоклассовой классификации

---

## 3. Forward Pass

1. Вход → слой
2. Преобразование → выход
3. Повторение через все слои → финальный результат

Математически:

```
y_pred = f_L(... f_2(f_1(x)) ...)
```

---

## 4. Backpropagation

* Алгоритм обучения нейросети
* Вычисляет градиенты loss по весам
* Обновление весов через оптимизатор (SGD, Adam)

```
W = W - lr * dL/dW
```

---

## 5. Loss Functions

* Regression: MSE, MAE
* Classification: Cross-Entropy
* Generative: KL-divergence, Negative Log-Likelihood

---

## 6. Optimizers

* SGD — простая стохастическая градиентная оптимизация
* Adam / AdamW — адаптивный learning rate
* RMSprop — стабилизирует обучение

---

## 7. Regularization

* Dropout — отключение нейронов для предотвращения overfitting
* Weight decay — штраф за большие веса
* Early stopping — остановка обучения при стагнации loss

---

## 8. Convolutional Neural Networks (CNN)

* Для изображений и сигналов
* Фильтры (kernels) → извлекают признаки
* Pooling → уменьшение размерности

---

## 9. Recurrent Neural Networks (RNN)

* Для последовательностей: текст, время
* LSTM / GRU решают проблему затухающего градиента
* Позволяют «помнить» прошлый контекст

---

## 10. Transformers

* Используют self-attention для глобального контекста
* Масштабируемы на большие тексты
* Основа современных LLM (GPT, Qwen, LLaMA)

---

## 11. Training Pipeline

1. Data preprocessing
2. Batch splitting
3. Forward pass
4. Loss computation
5. Backpropagation
6. Optimizer step
7. Validation & metrics
8. Repeat for epochs

---

## 12. Deep Learning vs Classic ML

| Classic ML                 | Deep Learning                       |
| -------------------------- | ----------------------------------- |
| Ручное feature engineering | Автоматическое извлечение признаков |
| Малые и средние данные     | Большие наборы данных               |
| Простые модели             | Многослойные, сложные модели        |
| Быстрое обучение           | Дорого по ресурсам                  |
| Интерпретируемость         | Сложно интерпретировать             |

---

## 13. Интуитивно

Deep Learning = «многоуровневая система обработки информации», где каждый слой автоматически выучивает признаки, а верхние слои объединяют их для сложных задач.

* Вход → низкоуровневые признаки → высокоуровневые представления → решение задачи

---

# Краткое резюме

Deep Learning — основа современных LLM и AI:

* Многослойные нейронные сети
* Backpropagation и оптимизация
* Регуляризация для борьбы с переобучением
* CNN, RNN, Transformer — разные типы архитектур для разных данных
* Позволяет модели **сама учиться признакам**, без ручной инженерии
