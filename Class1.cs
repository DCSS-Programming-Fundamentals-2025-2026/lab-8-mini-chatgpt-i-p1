using CommandLine;

public class Options
{
    [Option('d', "data", Default = "data/sample.txt", HelpText = "Шлях до файлу корпусу")]
    public string Data { get; set; }

    [Option('m', "model", Default = "trigram", HelpText = "Тип моделі: bigram, trigram, tinynn, tinytransformer")]
    public string Model { get; set; }

    [Option('t', "tokenizer", Default = "word", HelpText = "Токенізатор: word, char")]
    public string Tokenizer { get; set; }

    [Option('e', "epochs", Default = 3, HelpText = "Кількість епох тренування")]
    public int Epochs { get; set; }

    [Option('o', "out", Default = "checkpoint.json", HelpText = "Шлях для збереження checkpoint")]
    public string Out { get; set; }

    [Option('s', "seed", Default = 42, HelpText = "Seed для відтворюваності")]
    public int Seed { get; set; }

    [Option('l', "lr", Default = 0.1, HelpText = "Learning rate (для TinyNN)")]
    public double LearningRate { get; set; }
}
