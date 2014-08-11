package ml.shifu.core.request;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.RequestProcessService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RequestDispatcher {
    private static final Logger log = LoggerFactory.getLogger(RequestDispatcher.class);

    private RequestDispatcher() {
    }

    public static void dispatch(Request req) throws Exception {

        log.info("Starting Processor ...");

        SimpleModule module = new SimpleModule();
        module.set(req);

        Injector injector = Guice.createInjector(module);

        RequestProcessService service = injector.getInstance(RequestProcessService.class);
        service.exec(req);

    }
}
