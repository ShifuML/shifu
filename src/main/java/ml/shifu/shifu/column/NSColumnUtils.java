package ml.shifu.shifu.column;

/**
 * Created by zhanhu on 3/23/17.
 */
public class NSColumnUtils {

    public static boolean isColumnEqual(String fullColumnNameA, String fullColumnNameB) {
        NSColumn colmnA = new NSColumn(fullColumnNameA);
        NSColumn colmnB = new NSColumn(fullColumnNameB);
        return colmnA.equals(colmnB);
    }
}
