namespace PricePrediction
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Text;
    using Microsoft.ML;
    using Newtonsoft.Json;

    public static class Program
    {
        private const string DataFilePath = @"..\..\..\carsbg.csv";
        private const string MLModelFile = @"..\..\..\MLModel.zip";

        public static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;

            if (!File.Exists(MLModelFile))
            {
                TrainModel(DataFilePath, MLModelFile);
            }

            var testModelData = new List<ModelInput>
                                {
                                    new ModelInput
                                    {
                                        Make = "VW",
                                        Model = "Golf",
                                        CubicCapacity = 1400,
                                        FuelType = "Petrol",
                                        Gear = "Manual",
                                        HorsePower = 60,
                                        Range = 200000,
                                        Year = "1992"
                                    },
                                };

            testModelData.Dump();

            TestModel(MLModelFile, testModelData);
        }

        private static void TestModel(string modelFile, IEnumerable<ModelInput> testModelData)
        {
            var context = new MLContext();
            var model = context.Model.Load(modelFile, out _);

            var predictionEngine = context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            foreach (var testData in testModelData)
            {
                var prediction = predictionEngine.Predict(testData);
                Console.WriteLine(new string('-', 60));
                Console.WriteLine($"Input: {testData.Dump()}");
                Console.WriteLine($"Prediction: {prediction.Score}");
            }
        }

        private static void TrainModel(string dataFilePath, string mlModelFile)
        {
            var context = new MLContext();

            IDataView trainingDataView = context.Data.LoadFromTextFile<ModelInput>(
                path: dataFilePath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);

            var dataProcessPipeline = context.Transforms.Categorical.OneHotEncoding
                    (
                        new[] 
                        { 
                            new InputOutputColumnPair("Make", "Make"), 
                            new InputOutputColumnPair("FuelType", "FuelType"), 
                            new InputOutputColumnPair("Year", "Year"), 
                            new InputOutputColumnPair("Gear", "Gear")
                        })
                .Append(context.Transforms.Categorical.OneHotHashEncoding(
                    new[]
                    {
                        new InputOutputColumnPair("Model", "Model")
                    }))
                .Append(context.Transforms.Concatenate("Features", 
                    new[]
                    {
                        "Make", "FuelType", "Year", "Gear", "Model", "HorsePower", "Range", "CubicCapacity"
                    }));

            var trainer = context.Regression.Trainers.FastTree(labelColumnName: "Price", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer model = trainingPipeline.Fit(trainingDataView);
            context.Model.Save(model, trainingDataView.Schema, mlModelFile);
        }

        private static string Dump(this object obj)
        {
            return JsonConvert.SerializeObject(obj, Formatting.None);
        }
    }
}
