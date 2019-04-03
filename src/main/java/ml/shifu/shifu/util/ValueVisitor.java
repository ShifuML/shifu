package ml.shifu.shifu.util;

/**
 * Created by zhanhu on 4/9/18.
 */
public interface ValueVisitor {

    public void inject(Object key, Object value);

}
