package ml.shifu.shifu.request;


import ml.shifu.shifu.container.ShifuRequest;
import ml.shifu.shifu.request.processor.*;


public class RequestDispatcher {

    public static void dispatch(ShifuRequest shifuRequest) throws Exception {
        for (RequestObject req : shifuRequest.getRequests()) {
            dispatch(req);
        }
    }

    public static void dispatch(RequestObject req) throws Exception {
        String requestType = req.getRequestType();

        if (requestType.equalsIgnoreCase("CreateDataDictionary")) {
            CreateDataDictionaryRequestProcessor.run(req);
        } else if (requestType.equalsIgnoreCase("CreateModelElement")) {
            CreateModelElementRequestProcessor.run(req);
        } else if (requestType.equalsIgnoreCase("CreateMiningSchema")) {
            CreateMiningSchemaRequestProcessor.run(req);
        } else if (requestType.equalsIgnoreCase("ExecStats")) {
            ExecStatsRequestProcessor processor = new ExecStatsRequestProcessor();
            processor.run(req);
        } else if (requestType.equalsIgnoreCase("UpdateMiningSchema")) {
            RequestProcessor processor = new UpdateMiningSchemaRequestProcessor();
            processor.run(req);
        } else if (requestType.equalsIgnoreCase("CreateLocalTransformations")) {
            CreateLocalTransformationsRequestProcessor processor = new CreateLocalTransformationsRequestProcessor();
            processor.run(req);
        } else if (requestType.equalsIgnoreCase("ExecTransform")) {
            ExecTransformRequestProcessor processor = new ExecTransformRequestProcessor();
            processor.run(req);
        } else if (requestType.equalsIgnoreCase("ExecTrain")) {
            TrainingRequestProcessor processor = new TrainingRequestProcessor();
            processor.run(req);
        } else {
            throw new RuntimeException("Not a valid RequestType: " + requestType);
        }
    }

}
