package ml.shifu.shifu.request;


import ml.shifu.shifu.request.processor.CreateDataDictionaryRequestProcessor;

public class RequestDispatcher {

    public void dispatch(RequestObject req) {
        String action = req.getRequestType();

        if (action.equalsIgnoreCase("init")) {
            CreateDataDictionaryRequestProcessor processor = new CreateDataDictionaryRequestProcessor();
            processor.run(req);
        } else if (action.equalsIgnoreCase("stats")) {

        } else {
            throw new RuntimeException("Not a valid action: " + action);
        }
    }

}
