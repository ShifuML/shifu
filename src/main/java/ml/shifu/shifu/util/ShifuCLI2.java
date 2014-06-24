package ml.shifu.shifu.util;


import ml.shifu.shifu.container.ShifuRequest;
import ml.shifu.shifu.request.RequestDispatcher;

import java.io.File;

public class ShifuCLI2 {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.exit(0);
        }

        File reqFile = new File(args[0]);
        ShifuRequest req = JSONUtils.readValue(reqFile, ShifuRequest.class);
        RequestDispatcher.dispatch(req);

    }

}
