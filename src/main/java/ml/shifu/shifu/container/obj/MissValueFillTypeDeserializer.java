package ml.shifu.shifu.container.obj;

import java.io.IOException;

import ml.shifu.shifu.container.obj.ModelNormalizeConf.MissValueFillType;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;

/**
 * To deserialize {@link MissValueFillType} instance.
 * 
 * @author xiaobzheng (zheng.xiaobin.roubao@gmail.com)
 *
 */
public class MissValueFillTypeDeserializer extends JsonDeserializer<MissValueFillType> {

    /*
     * (non-Javadoc)
     * 
     * @see com.fasterxml.jackson.databind.JsonDeserializer#deserialize(com.fasterxml.jackson.core.JsonParser,
     * com.fasterxml.jackson.databind.DeserializationContext)
     */
    @Override
    public MissValueFillType deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException, JsonProcessingException {
        ObjectCodec oc = jp.getCodec();
        JsonNode node = oc.readTree(jp);

        for(MissValueFillType value: MissValueFillType.values()) {
            if(value.name().equalsIgnoreCase(node.textValue())) {
                return value;
            }
        }
        return null;
    }
}
