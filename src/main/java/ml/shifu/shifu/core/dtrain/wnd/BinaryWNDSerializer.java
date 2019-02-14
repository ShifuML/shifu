package ml.shifu.shifu.core.dtrain.wnd;

import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.zip.GZIPOutputStream;

/**
 * Binary IndependentWNDModel serializer.
 *
 * @author: Wu Devin (haifwu@paypal.com)
 */
public class BinaryWNDSerializer {
    public static void save(ModelConfig modelConfig, WideAndDeep wideAndDeep, FileSystem fs, Path output)
            throws IOException {
        DataOutputStream dos = null;
        try {
            dos = new DataOutputStream(new GZIPOutputStream(fs.create(output)));
            // version
            dos.writeInt(CommonConstants.WND_FORMAT_VERSION);
            dos.writeUTF(modelConfig.getAlgorithm());
            PersistWideAndDeep.save(wideAndDeep, dos);
            dos.writeUTF(modelConfig.getNormalizeType().name());
            dos.writeDouble(modelConfig.getNormalizeStdDevCutOff());
        } finally {
            IOUtils.closeStream(dos);
        }
    }
}
