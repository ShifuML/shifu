package ml.shifu.core.di.service;


import com.google.inject.Inject;
import ml.shifu.core.di.spi.RequestDispatcher;
import ml.shifu.core.request.Request;

public class RequestDispatchService {

    @Inject
    private RequestDispatcher dispatcher;

    public void dispatch(Request req) throws Exception {
        dispatcher.dispatch(req);
    };


}
