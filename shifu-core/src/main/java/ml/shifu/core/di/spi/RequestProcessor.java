package ml.shifu.core.di.spi;

import ml.shifu.core.request.Request;

public interface RequestProcessor {

    public void exec(Request req) throws Exception;

}
