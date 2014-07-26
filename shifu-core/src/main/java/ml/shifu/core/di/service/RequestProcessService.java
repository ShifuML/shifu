package ml.shifu.core.di.service;

import com.google.inject.Inject;
import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;

public class RequestProcessService {

    @Inject
    private RequestProcessor processor;

    public void exec(Request req) throws Exception {
        processor.exec(req);
    }

}
