package ml.shifu.core.request.processor;

import ml.shifu.core.request.RequestObject;

public interface RequestProcessor {

    public void run(RequestObject req) throws Exception;

}
