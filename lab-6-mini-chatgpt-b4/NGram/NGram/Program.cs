using System.Text.Json;
using Lib.Tokenization;
using Lib.Tokenization.Serialization;
using NGram;
using NGram.ModelFactory;
using NGram.Metrics;

Console.OutputEncoding = System.Text.Encoding.UTF8;

Console.WriteLine("--- Mini ChatGPT — NGram Pipeline ---");
Console.WriteLine();

Console.WriteLine("Крок 1: Завантаження корпусу...");

CorpusClass corpus = CorpusLoader.Load("data/showcase.txt");

Console.WriteLine($"  TrainText: {corpus.TrainText!.Length} символів");
Console.WriteLine($"  ValText:   {corpus.ValText!.Length} символів");
Console.WriteLine();

Console.WriteLine("Крок 2: Побудова WordTokenizer...");

WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);

Console.WriteLine($"  VocabSize: {tokenizer.VocabSize} слів");
Console.WriteLine();

Console.WriteLine("Крок 3: Токенізація тексту...");

int[] trainTokens = tokenizer.Encode(corpus.TrainText);
int[] valTokens   = tokenizer.Encode(corpus.ValText);

Console.WriteLine($"  Train tokens: {trainTokens.Length}");
Console.WriteLine($"  Val tokens:   {valTokens.Length}");
Console.WriteLine();

Console.WriteLine("Крок 4: Тренування моделей...");

NGramModelFactory factory = new NGramModelFactory();

NGramModel bigramModel = factory.CreateBigramModel(tokenizer.VocabSize);
bigramModel.Train(trainTokens);
Console.WriteLine("  Bigram модель натренована");

TrigramModel trigramModel = factory.CreateTrigramModel(tokenizer.VocabSize);
trigramModel.Train(trainTokens);
Console.WriteLine("  Trigram модель натренована");
Console.WriteLine();

Console.WriteLine("Крок 5: Обчислення Perplexity...");

double bigramPP  = PerplexityCalculator.ComputePerplexity(
    trainTokens, ctx => bigramModel.NextTokenScores(ctx));
double trigramPP = PerplexityCalculator.ComputePerplexity(
    trainTokens, ctx => trigramModel.NextTokenScores(ctx));
double trigramValPP = PerplexityCalculator.ComputePerplexity(
    valTokens, ctx => trigramModel.NextTokenScores(ctx));

Console.WriteLine($"  Bigram  (train) : {bigramPP:F2}");
Console.WriteLine($"  Trigram (train) : {trigramPP:F2}");
Console.WriteLine($"  Trigram (val)   : {trigramValPP:F2}");

if (trigramPP < bigramPP)
    Console.WriteLine("  Trigram < Bigram — очікувано!");
Console.WriteLine();

Console.WriteLine("Крок 6: Генерація тексту (Greedy)...");

var context = new List<int> { trainTokens[0], trainTokens[1] };
for (int i = 0; i < 20; i++)
{
    float[] scores = trigramModel.NextTokenScores(context.ToArray());
    int next = Array.IndexOf(scores, scores.Max());
    context.Add(next);
}

string generatedText = tokenizer.Decode(context.ToArray());
Console.WriteLine($"  Згенерований текст:");
Console.WriteLine($"  \"{generatedText}\"");
Console.WriteLine();

Console.WriteLine("Крок 7: Збереження Checkpoint...");

string checkpointDir = "checkpoint";
Directory.CreateDirectory(checkpointDir);

NGramPayloadMapper modelPayload = trigramModel.GetPayloadForCheckpoint();
string modelJson = JsonSerializer.Serialize(modelPayload, new JsonSerializerOptions { WriteIndented = false });
File.WriteAllText(Path.Combine(checkpointDir, "checkpoint_model.json"), modelJson);
Console.WriteLine("  checkpoint/checkpoint_model.json збережено");

object tokenizerPayload = tokenizer.GetPayloadForCheckpoint();
string tokenizerJson = JsonSerializer.Serialize(tokenizerPayload);
File.WriteAllText(Path.Combine(checkpointDir, "checkpoint_tokenizer.json"), tokenizerJson);
Console.WriteLine("  checkpoint/checkpoint_tokenizer.json збережено");
Console.WriteLine();

Console.WriteLine("Крок 8: Відновлення з Checkpoint...");

string loadedModelJson = File.ReadAllText(Path.Combine(checkpointDir, "checkpoint_model.json"));
TrigramModel restoredModel = new TrigramModel(tokenizer.VocabSize);
restoredModel.FromPayload(JsonDocument.Parse(loadedModelJson).RootElement);
Console.WriteLine("  Trigram модель відновлена");

string loadedTokenizerJson = File.ReadAllText(Path.Combine(checkpointDir, "checkpoint_tokenizer.json"));
JsonElement tokPayloadElement = JsonDocument.Parse(loadedTokenizerJson).RootElement;
var restoredVocab = TokenizerPayloadSerializer.DeserializeWordVocab(tokPayloadElement);
WordTokenizer restoredTokenizer = new WordTokenizer(restoredVocab);
Console.WriteLine("  WordTokenizer відновлений");

var context2 = new List<int> { trainTokens[0], trainTokens[1] };
for (int i = 0; i < 20; i++)
{
    float[] scores = restoredModel.NextTokenScores(context2.ToArray());
    int next = Array.IndexOf(scores, scores.Max());
    context2.Add(next);
}

string restoredText = restoredTokenizer.Decode(context2.ToArray());
Console.WriteLine($"  Текст після відновлення:");
Console.WriteLine($"  \"{restoredText}\"");
Console.WriteLine();

bool isMatch = generatedText == restoredText;
Console.WriteLine(isMatch
    ? "  Checkpoint працює коректно — тексти ідентичні!"
    : "  ПОМИЛКА — тексти не збігаються!");
Console.WriteLine();

Console.WriteLine("╔══════════════════════════════════════════════╗");
Console.WriteLine("║              Підсумок Pipeline               ║");
Console.WriteLine("╠══════════════════════════════════════════════╣");
Console.WriteLine($"║  Корпус          : showcase.txt              ║");
Console.WriteLine($"║  VocabSize       : {tokenizer.VocabSize,-25}║");
Console.WriteLine($"║  Train tokens    : {trainTokens.Length,-25}║");
Console.WriteLine($"║  Val tokens      : {valTokens.Length,-25}║");
Console.WriteLine($"║  Bigram PP       : {bigramPP,-25:F2}║");
Console.WriteLine($"║  Trigram PP      : {trigramPP,-25:F2}║");
Console.WriteLine($"║  Checkpoint OK   : {(isMatch ? "Так" : "Ні"),-25}║");
Console.WriteLine("╚══════════════════════════════════════════════╝");
