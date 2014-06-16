package ml.shifu.shifu.request.processor;

import ml.shifu.shifu.request.RequestObject;

public interface RequestProcessor {

    public void run(RequestObject req) throws Exception;

}
