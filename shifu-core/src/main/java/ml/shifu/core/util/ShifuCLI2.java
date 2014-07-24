package ml.shifu.core.util;


import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.container.ShifuRequest;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.RequestDispatchService;
import ml.shifu.core.request.Request;
import ml.shifu.core.request.RequestDispatcher;


import java.io.File;

public class ShifuCLI2 {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.exit(0);
        }

        File reqFile = new File(args[0]);
        Request req = JSONUtils.readValue(reqFile, Request.class);
        RequestDispatcher.dispatch(req);

        //SimpleModule module = new SimpleModule();
        //module.set(req);
        //Injector injector = Guice.createInjector(module);
        //RequestDispatchService service = injector.getInstance(RequestDispatchService.class);
        //service.dispatch(req);
    }

}
