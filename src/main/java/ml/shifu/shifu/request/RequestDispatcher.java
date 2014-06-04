package ml.shifu.shifu.request;


import ml.shifu.shifu.request.processor.InitRequestProcessor;

public class RequestDispatcher {

    public void dispatch(RequestObject req) {
        String action = req.getAction();

        if (action.equalsIgnoreCase("init")) {
            InitRequestProcessor processor = new InitRequestProcessor();
            processor.run(req);
        } else if (action.equalsIgnoreCase("stats")) {

        } else {
            throw new RuntimeException("Not a valid action: " + action);
        }
    }

}
