# Baseline End-to-End Integration Guide (Stage 3)

Цей документ описує як підключити бібліотеки **Corpus**, **Tokenization** та **NGram**
у єдиний пайплайн для Trainer CLI та Chat.

---

## 1. Підключення залежностей

Додати у `.csproj` вашого проекту:

```xml
<ItemGroup>
  <ProjectReference Include="..\Lib.Corpus\Lib.Corpus.csproj" />
  <ProjectReference Include="..\Lib.Tokenization\Lib.Tokenization.csproj" />
  <ProjectReference Include="..\NGram\NGram.csproj" />
</ItemGroup>
```

Using директиви у коді:

```csharp
using Lib.Tokenization;
using NGram;
using NGram.ModelFactory;
using NGram.Metrics;
```

---

## 2. Повний пайплайн: текст → генерація

```csharp
// Крок 1: Завантаження корпусу
var corpus = CorpusLoader.Load("data/showcase.txt");
// corpus.TrainText — 90% тексту для тренування
// corpus.ValText   — 10% тексту для валідації

// Крок 2: Побудова токенізатора
var tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
// tokenizer.VocabSize — розмір словника

// Крок 3: Токенізація
int[] trainTokens = tokenizer.Encode(corpus.TrainText);
int[] valTokens   = tokenizer.Encode(corpus.ValText);

// Крок 4: Тренування моделі
var factory = new NGramModelFactory();
var model = factory.CreateTrigramModel(tokenizer.VocabSize);
model.Train(trainTokens);

// Крок 5: Greedy generation loop
var context = new List<int> { trainTokens[0], trainTokens[1] };
for (int i = 0; i < 20; i++)
{
    float[] scores = model.NextTokenScores(context.ToArray());
    int next = Array.IndexOf(scores, scores.Max());
    context.Add(next);
}

// Крок 6: Декодування
string text = tokenizer.Decode(context.ToArray());
Console.WriteLine(text); // "калина похилилася журиться..."
```

---

## 3. Формат Checkpoint (Save / Load)

### Збереження

```csharp
// Модель
var modelPayload = model.GetPayloadForCheckpoint();
string modelJson = JsonSerializer.Serialize(modelPayload);
File.WriteAllText("checkpoint_model.json", modelJson);

// Токенізатор
var tokenizerPayload = tokenizer.GetPayloadForCheckpoint();
string tokenizerJson = JsonSerializer.Serialize(tokenizerPayload);
File.WriteAllText("checkpoint_tokenizer.json", tokenizerJson);
```

`NGramPayloadMapper` (для Trigram):
```json
{
  "BigramProbs": [[0.1, 0.9, ...], ...],
  "TrigramProbs": [
    { "Prev2": 0, "Prev1": 1, "NextTokenScores": [0.0, 0.5, 0.5, ...] },
    ...
  ]
}
```

`WordTokenizer` payload:
```json
{
  "words": ["<UNK>", "калина", "похилилася", "вітер", ...]
}
```

### Відновлення

```csharp
// Відновлення моделі
string modelJson = File.ReadAllText("checkpoint_model.json");
var restoredModel = new TrigramModel(vocabSize);
restoredModel.FromPayload(JsonDocument.Parse(modelJson).RootElement);

// Відновлення токенізатора
string tokenizerJson = File.ReadAllText("checkpoint_tokenizer.json");
var payload = JsonDocument.Parse(tokenizerJson).RootElement;
var vocab = TokenizerPayloadSerializer.DeserializeWordVocab(payload);
var restoredTokenizer = new WordTokenizer(vocab);
```

---

## 4. Greedy Generation Loop для Chat

```csharp
// Початковий контекст (перші два токени)
var context = new List<int> { startToken1, startToken2 };

for (int step = 0; step < maxSteps; step++)
{
    float[] scores = model.NextTokenScores(context.ToArray());

    // Greedy: обираємо токен з максимальною ймовірністю
    int nextToken = Array.IndexOf(scores, scores.Max());

    context.Add(nextToken);
}

string result = tokenizer.Decode(context.ToArray());
```

> **Примітка:** Greedy вибір детермінований — однаковий контекст завжди дасть однаковий результат.
> Для різноманітності виведення можна використати **temperature sampling** (Stage 3, Lib.Sampling).

---

## 5. Word vs Char — яку гранулярність обрати?

- **Word**: 1 крок = 1 слово, словник ~сотні слів, добре на малих даних
- **Char**: 1 крок = 1 символ, словник ~50-100 символів, потребує більше даних

**Що обрати:** для малого корпусу (~300 слів) — Word. Для великого (10k+ символів) — Char.

---

## 6. Perplexity як метрика якості

```csharp
// Обчислення perplexity на валідаційних даних
double pp = PerplexityCalculator.ComputePerplexity(
    valTokens,
    ctx => model.NextTokenScores(ctx)
);

Console.WriteLine($"Perplexity: {pp:F2}");
// Менше — краще. Trigram < Bigram на тренувальних даних.
```

Порівняння моделей:

```csharp
double bigramPP  = PerplexityCalculator.ComputePerplexity(trainTokens,
    ctx => bigramModel.NextTokenScores(ctx));
double trigramPP = PerplexityCalculator.ComputePerplexity(trainTokens,
    ctx => trigramModel.NextTokenScores(ctx));

// trigramPP < bigramPP — trigram краще запам'ятовує тренувальні патерни
```

---

## 7. Інтеграція у Trainer CLI (Stage 3)

```
args: --model trigram --tokenizer word --data data/showcase.txt --output checkpoint/
```

Кроки у Trainer:
1. `CorpusLoader.Load(args.Data)` → `CorpusClass`
2. `WordTokenizer.BuildFromText(corpus.TrainText)` → `ITokenizer`
3. `tokenizer.Encode(corpus.TrainText)` → `int[]`
4. `NGramModelFactory.CreateTrigramModel(tokenizer.VocabSize)` → `TrigramModel`
5. `model.Train(tokens)`
6. Серіалізація → `checkpoint/`
