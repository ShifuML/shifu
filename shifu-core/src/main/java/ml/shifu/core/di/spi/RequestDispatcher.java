package ml.shifu.core.di.spi;

import ml.shifu.core.request.Request;

public interface RequestDispatcher {

    public void dispatch(Request req) throws Exception;

}
