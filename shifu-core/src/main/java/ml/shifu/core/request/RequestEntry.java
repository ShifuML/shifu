package ml.shifu.core.request;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.RequestDispatchService;

public class RequestEntry {

    public static void run(Request req) throws Exception {
        SimpleModule module = new SimpleModule();
        module.set(req);
        Injector injector = Guice.createInjector(module);
        RequestDispatchService service = injector.getInstance(RequestDispatchService.class);
        service.dispatch(req);
    }
}
