using CommandLine;
using Lib.Corpus.Configuration;
using Lib.Corpus.Domain;
using Lib.Corpus.Infrastructure;
using Lib.Corpus.Processing;
using Lib.Tokenization.Interfaces;
using Lib.Tokenization.Model;
using Lib.Tokenization.Tokanizers;
using System.Net.Security;

namespace Trainer;

class Trainer
{
    public static void Main(string[] args)
    {
        string inputCommands = Console.ReadLine();
        Parser.Default.ParseArguments<Options>(args).WithParsed(RunOptions).WithNotParsed(HandleParseError);

        //var splitter = new CorpusSplitter();
        //var normilizer = new CorpusTextNormalizer();
        //var fileSystem = new DefaultFileSystem();
        //var loader = new CorpusLoader(normilizer, splitter, fileSystem);

        //string dataPath = Path.Combine("data", "Example.txt");

        //if (!File.Exists(dataPath))
        //{
        //    Console.WriteLine("Файл не знайдено");
        //    return;
        //}

        //CorpusLoadOptions loadOptions = new CorpusLoadOptions();
        //var corpus = loader.Load(dataPath, loadOptions);

        //Console.WriteLine($"Корпус довжиною --{corpus?.TrainText.Length} Завантажено");

        //ITokenizerFactory tokenizerFactory = new WordTokenizerFactory();

        //ITokenizer tokenizer = tokenizerFactory.BuildFromText(corpus.TrainText);
        //int[] codedTrainTokens = tokenizer.Encode(corpus.TrainText);

        //Console.WriteLine($"[B1] Токенізація успішна. Розмір словника: {tokenizer.VocabSize}");
        //Console.WriteLine($"[B1] Всього отримано токенів: {codedTrainTokens.Length}");

        //Тут продовжує команда Б2
    }

    public static void RunOptions(Options opts)
    {
        Console.WriteLine($"Your model: {opts.Model}");

        var splitter = new CorpusSplitter();
        var normilizer = new CorpusTextNormalizer();
        var fileSystem = new DefaultFileSystem();
        var loader = new CorpusLoader(normilizer, splitter, fileSystem);

        string dataPath = opts.Data;

        if (!File.Exists(dataPath))
        {
            Console.WriteLine("Файл не знайдено");
            return;
        }
    }
}



