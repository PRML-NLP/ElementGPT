import re

def process_child_growth(doc):
    """우리아이 성장백과 document preprocessing"""
    
    parsed = []
    heading = False
    concat = False
    try:
        for span in doc:
            class_names = span.attrs.get("class")
            if class_names:
                if "se_fs_T4" in class_names:
                    # ignore and append to next line
                    concat = True
                    if parsed:
                        parsed[-1] += span.text
                    else:
                        parsed.append(span.text)
                    
                elif not heading and "se_fs_T2" in class_names:
                    parsed.append("### " + span.text)
                    heading = True
                else:
                    if re.compile(r"[가-힣]+").match(span.text):
                        parsed[-1] += span.text
            else:
                heading = False
                # if span.text is composed of korean characters'
                if re.compile(r"[가-힣]+").match(span.text):
                    if concat:
                        parsed[-1] += span.text
                        concat = False
                    else:
                        parsed.append(span.text)
                elif re.compile(r"[1-9].").match(span.text):
                    concat = True
                    parsed.append(span.text)
                    
    except Exception as e:
        print(parsed)
        print(heading)
        print(span)
        
    return parsed