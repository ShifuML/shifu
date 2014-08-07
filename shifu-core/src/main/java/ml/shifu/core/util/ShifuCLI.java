package ml.shifu.core.util;


import ml.shifu.core.request.Request;
import ml.shifu.core.request.RequestDispatcher;

import java.io.File;

public class ShifuCLI {

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
