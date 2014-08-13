package ml.shifu.core.request;

import ml.shifu.core.util.JSONUtils;

import java.io.File;

public class RequestDispatcherTest {


    //@Test
    public void test1() throws Exception {
        Request req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/requests/1_create.json"), Request.class);
        RequestDispatcher.dispatch(req);
    }

    //@Test
    public void test2() throws Exception {
        Request req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/requests/2_stats.json"), Request.class);
        RequestDispatcher.dispatch(req);
    }

    //@Test
    public void test3() throws Exception {
        Request req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/new/requests/3_varselect.json"), Request.class);
        RequestDispatcher.dispatch(req);
    }

    //@Test
    public void test4() throws Exception {
        Request req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/new/requests/4_transformprep.json"), Request.class);
        RequestDispatcher.dispatch(req);
    }

    //@Test
    public void test5() throws Exception {
        Request req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/new/requests/5_transformexec.json"), Request.class);
        RequestDispatcher.dispatch(req);
    }

    //@Test
    public void test7() throws Exception {
        Request req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/new/requests/7_modelexec.json"), Request.class);
        RequestDispatcher.dispatch(req);
    }

    //@Test
    public void test8() throws Exception {
        Request req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/new/requests/8_modeleval.json"), Request.class);
        RequestDispatcher.dispatch(req);
    }
      /*
    //@Test
    public void test4() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("shifu-core/src/test/resources/models/wdbc/All/requests/4_ExecStatsRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }

    //@Test
    public void test45() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("shifu-core/src/test/resources/models/wdbc/All/requests/45_UpdateMiningSchemaRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }

    @Test
    public void test5() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/5_CreateLocalTransformationsRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }

    //@Test
    public void test6() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("shifu-core/src/test/resources/models/wdbc/All/requests/6_ExecTransformRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);

    }

    //@Test
    public void test9() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/9_ExecModelRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);

    }

    //@Test
    public void test10() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/All/requests/10_ModelEvaluationRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);

    }           */


}
