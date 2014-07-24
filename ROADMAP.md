


* CreatePMMLElementRequest
    * [YES] DataDictionaryCreator
    * [NO] TransformationDictionaryCreator
    * [YES] TargetsCreator
    * [NO] OutputCreator
    * [YES] MiningSchemaCreator
    * [YES] DerivedFieldCreator
    * [NO] ModelExplaination
    * [NO] ModelVerification
* CalcStatsRequest
    * [YES] UnivariateStatsCalculator
    * [NO] MultivariateStatsCalculator
    * [NO] ANOVACalculator
* TransformRequest

* UpdateMiningSchemaRequest

* TrainRequest

* ModelExecRequest

* ModelEvalRequest


NamePattern
* [YES] $default
* [YES] exact-match
* [NO] pattern matching


FileLoading
    * [YES] Local
    * [NO] Akka


Finish the list: PMMLUtils

TODO: MiningSchema -> Transformation Map

injectable: data loader