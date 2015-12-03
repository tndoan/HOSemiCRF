package ChineseSegPOS;


public class WordDetails {

    String[] elements;

    public WordDetails() {
        elements = null;
    }

    // Exclude the last token, it is the label
    public WordDetails(String[] toks) {
        elements = new String[toks.length-1];
        for (int i=0; i<toks.length-1; i++) {
            elements[i] = toks[i];
        }
    }

    public String getElement(int col) {
        if (col >= 0 && col < elements.length) {
            return elements[col];
        } else {
            return "";
        }
    }

    @Override
    public String toString() {
        String str = "";
        for (String element : elements) {
            str += element + " ";
        }
        return str.trim();
    }
}

