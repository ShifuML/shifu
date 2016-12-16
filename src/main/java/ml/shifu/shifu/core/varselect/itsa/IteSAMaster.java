package ml.shifu.shifu.core.varselect.itsa;

import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.core.dtrain.nn.NNMaster;

/**
 * Created by zhanhu on 11/7/16.
 */
public class IteSAMaster extends AbstractMasterComputable<MasterIteSAParams, WorkerIteSAParams> {

    private NNMaster master;

    @Override
    public void init(MasterContext<MasterIteSAParams, WorkerIteSAParams> context) {
        master = new NNMaster();
        master.init(null);
    }



    @Override
    public MasterIteSAParams doCompute(MasterContext<MasterIteSAParams, WorkerIteSAParams> context) {
        return null;
    }

}
