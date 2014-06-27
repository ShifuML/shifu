package ml.shifu.core.request;

import ml.shifu.core.container.ShifuRequest;
import ml.shifu.core.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;

public class RequestDispatcherTest {

    @Test
    public void test1() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/1_CreateDictionaryRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }

    @Test
    public void test2() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/2_CreateModelElementRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }

    @Test
    public void test3() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/3_CreateMiningSchemaRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }

    @Test
    public void test4() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/4_ExecStatsRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }

    @Test
    public void test45() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/45_UpdateMiningSchemaRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }

    @Test
    public void test5() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/5_CreateLocalTransformationsRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }

    @Test
    public void test6() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/6_ExecTransformRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);

    }

    @Test
    public void test8() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/8_ExecTrainRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);

    }
}
