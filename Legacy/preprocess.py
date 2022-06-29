import numpy as np
import pandas as pd
import tokenize
from gensim.models import KeyedVectors


###############################################################################################################
##                                                                                                           ##
##                                                LEGACY CODE                                                ##
##                                                                                                           ##
###############################################################################################################


def seperate(string1):
    index = ''
    word = ''

    lstt = string1.split('##')
    for i in lstt[0]:
        if i.isdigit():
            index += i
        else: word += i
    word = word.replace('\t','')
    category = lstt[-1]

    return [index, word, category]


def embedding_nextword(a, b, vocabulary, model, zerolst):
    
    if (a[2] in vocabulary) and (b[2] in vocabulary):
        embedding1 = model.similarity(a[2], b[2])
        wordAfterunit1 = model[a[2]]
        wordAfterunit2 = model[b[2]]
    
    else:
        embedding1 = 0
        wordAfterunit1 = zerolst
        wordAfterunit2 = zerolst
    
    return embedding1, wordAfterunit1, wordAfterunit2


# needs embedding_nextword(...)
def pair(emb_filepath, lst):
    boundary_punctuation = ['！', '。', '……', '？']
    pairlst = []
    connective = ''
    templist = []

    model = KeyedVectors.load_word2vec_format(emb_filepath, binary=False)
    vocabulary = model.key_to_index

    for j in range(len(lst)):
        if lst[j][2] == 'Seg=B-Conn' or lst[j][1] in boundary_punctuation:
            connective += lst[j][1]
            index = lst[j][0]
            k = j+1
            while lst[k][2] == 'Seg=I-Conn':
                connective += lst[k][1]
                k += 1
            nextt = lst[k][1]
            templist.append([connective, index, nextt])
            connective = ''

    for a in range(len(templist) - 1):
        zerolst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if (templist[a][0] not in boundary_punctuation) and templist[a + 1][0] not in boundary_punctuation:
            
            distance = int(templist[a + 1][1]) - int(templist[a][1])
            
            if (templist[a][0] in vocabulary) and (templist[a + 1][0] in vocabulary):
                embedding = model.similarity(templist[a][0], templist[a + 1][0])
                unit1 = model[templist[a][0]]
                unit2 = model[templist[a + 1][0]]
                embedding1, wordAfterunit1, wordAfterunit2 = embedding_nextword(templist[a], templist[a + 1], vocabulary, model, zerolst)

            else:
                embedding = 0
                unit1 = zerolst
                unit2 = zerolst
                embedding1, wordAfterunit1, wordAfterunit2 = embedding_nextword(templist[a], templist[a + 1], vocabulary, model, zerolst)

            if distance > 0:
                pairlst.append(
                    [templist[a][0], templist[a + 1][0], (int(templist[a + 1][1]) - int(templist[a][1])), 0, embedding,
                     embedding1, unit1, unit2, wordAfterunit1, wordAfterunit2])  # intra-sentence

        elif templist[a][0] not in boundary_punctuation and templist[a + 1][0] in boundary_punctuation:
            aa = a + 1
            while aa < len(templist) and templist[aa][0] in boundary_punctuation:
                aa += 1
            if aa < len(templist):
                distance = int(templist[aa][1]) - int(templist[a][1])
                if (templist[a][0] in vocabulary) and (templist[aa][0] in vocabulary):
                    embedding = model.similarity(templist[a][0], templist[aa][0])
                    unit1 = model[templist[a][0]]
                    unit2 = model[templist[aa][0]]
                    embedding1, wordAfterunit1, wordAfterunit2 = embedding_nextword(templist[a], templist[aa], vocabulary, model, zerolst)

                else:
                    embedding = 0
                    unit1 = zerolst
                    unit2 = zerolst
                    embedding1, wordAfterunit1, wordAfterunit2 = embedding_nextword(templist[a], templist[aa], vocabulary, model, zerolst)

                if distance > 0:
                    pairlst.append([templist[a][0], templist[aa][0], (int(templist[aa][1]) - int(templist[a][1])),
                                    100, embedding, embedding1, unit1, unit2, wordAfterunit1, wordAfterunit2])  # cross-sentence

    return pairlst


# needs pair(lst) and seperate(string1)
def read_source(ctdb_filepath, emb_filepath):

    with tokenize.open(ctdb_filepath) as f:
        tokens = tokenize.generate_tokens(f.readline)

        string = ''
        for token in tokens:
            string += token.string

    seperateLst = []

    contents = string.replace('_______', '##').split('\n')

    for i in contents:
        word = seperate(i)
        bound = ['!', '。', '？', '……']
        if word[1] != '# newdoc id = chtb_':
            seperateLst.append(word)
    
    temp = ''
    templist = []

    for j in range(len(seperateLst)):
        if seperateLst[j][2] == 'Seg=B-Conn':
            temp += seperateLst[j][1]

            k = j + 1
            while seperateLst[k][2] == 'Seg=I-Conn':
                temp += seperateLst[k][1]
                k += 1

            templist.append(temp)
            temp = ''

    pairs = pair(emb_filepath, seperateLst)
    
    return pairs


###############################################################################################################
###############################################################################################################


def match(pairs, dictionary):
    '''
    Enrich... TBD
    '''

    # loop through entries in pairs
    for i in range(len(pairs)):
        # match dc from pairs with dictionary and append the manual annotation
        if pairs[i][0] == dictionary.iat[i,0] and pairs[i][1] == dictionary.iat[i,1] and dictionary.iat[i,2] == 'YES':
            pairs[i].append('YES')
        else:
            pairs[i].append('NO')

        # needs to be made more efficient - replace np.array entries for UnitA, UnitB, UnitANext and UnitBNext by their explicit entries
        for j in pairs[i][6]:
            pairs[i].append(j)
        for j in pairs[i][7]:
            pairs[i].append(j)
        for j in pairs[i][8]:
            pairs[i].append(j)
        for j in pairs[i][9]:
            pairs[i].append(j)
        del pairs[i][4:10]

    return pairs


# read excel sheets to a dictionary of dataframes for each sheet
dfs = pd.read_excel('Dictionaries/gold_standard.xlsx', sheet_name=None, header=None, names=['first_dc', 'second_dc', 'annotation'])

# list of dataset to create
ds_list = ['train', 'test', 'dev', 'devtest', 'cdtb']

# list of column names for each dataset
column_names = ['First_DC', 'Second_DC', 'Distance', 'Scenario', 'Annotation',
                 'UnitA1', 'UnitA2', 'UnitA3', 'UnitA4', 'UnitA5', 'UnitA6', 'UnitA7', 'UnitA8', 'UnitA9', 'UnitA10',
                 'UnitA11', 'UnitA12', 'UnitA13', 'UnitA14', 'UnitA15', 'UnitA16', 'UnitA17', 'UnitA18', 'UnitA19', 'UnitA20',
                 'UnitA21', 'UnitA22', 'UnitA23', 'UnitA24', 'UnitA25', 'UnitA26', 'UnitA27', 'UnitA28', 'UnitA29', 'UnitA30',
                 'UnitA31', 'UnitA32', 'UnitA33', 'UnitA34', 'UnitA35', 'UnitA36', 'UnitA37', 'UnitA38', 'UnitA39', 'UnitA40',
                 'UnitA41', 'UnitA42', 'UnitA43', 'UnitA44', 'UnitA45', 'UnitA46', 'UnitA47', 'UnitA48', 'UnitA49', 'UnitA50',
                 'UnitA51', 'UnitA52', 'UnitA53', 'UnitA54', 'UnitA55', 'UnitA56', 'UnitA57', 'UnitA58', 'UnitA59', 'UnitA60',
                 'UnitA61', 'UnitA62', 'UnitA63', 'UnitA64', 'UnitA65', 'UnitA66', 'UnitA67', 'UnitA68', 'UnitA69', 'UnitA70',
                 'UnitA71', 'UnitA72', 'UnitA73', 'UnitA74', 'UnitA75', 'UnitA76', 'UnitA77', 'UnitA78', 'UnitA79', 'UnitA80',
                 'UnitA81', 'UnitA82', 'UnitA83', 'UnitA84', 'UnitA85', 'UnitA86', 'UnitA87', 'UnitA88', 'UnitA89', 'UnitA90',
                 'UnitA91', 'UnitA92', 'UnitA93', 'UnitA94', 'UnitA95', 'UnitA96', 'UnitA97', 'UnitA98', 'UnitA99', 'UnitA100',
                 'UnitA101', 'UnitA102', 'UnitA103', 'UnitA104', 'UnitA105', 'UnitA106', 'UnitA107', 'UnitA108', 'UnitA109', 'UnitA110',
                 'UnitA111', 'UnitA112', 'UnitA113', 'UnitA114', 'UnitA115', 'UnitA116', 'UnitA117', 'UnitA118', 'UnitA119', 'UnitA120',
                 'UnitA121', 'UnitA122', 'UnitA123', 'UnitA124', 'UnitA125', 'UnitA126', 'UnitA127', 'UnitA128', 'UnitA129', 'UnitA130',
                 'UnitA131', 'UnitA132', 'UnitA133', 'UnitA134', 'UnitA135', 'UnitA136', 'UnitA137', 'UnitA138', 'UnitA139', 'UnitA140',
                 'UnitA141', 'UnitA142', 'UnitA143', 'UnitA144', 'UnitA145', 'UnitA146', 'UnitA147', 'UnitA148', 'UnitA149', 'UnitA150',
                 'UnitA151', 'UnitA152', 'UnitA153', 'UnitA154', 'UnitA155', 'UnitA156', 'UnitA157', 'UnitA158', 'UnitA159', 'UnitA160',
                 'UnitA161', 'UnitA162', 'UnitA163', 'UnitA164', 'UnitA165', 'UnitA166', 'UnitA167', 'UnitA168', 'UnitA169', 'UnitA170',
                 'UnitA171', 'UnitA172', 'UnitA173', 'UnitA174', 'UnitA175', 'UnitA176', 'UnitA177', 'UnitA178', 'UnitA179', 'UnitA180',
                 'UnitA181', 'UnitA182', 'UnitA183', 'UnitA184', 'UnitA185', 'UnitA186', 'UnitA187', 'UnitA188', 'UnitA189', 'UnitA190',
                 'UnitA191', 'UnitA192', 'UnitA193', 'UnitA194', 'UnitA195', 'UnitA196', 'UnitA197', 'UnitA198', 'UnitA199', 'UnitA200',
                 'UnitB1', 'UnitB2', 'UnitB3', 'UnitB4', 'UnitB5', 'UnitB6', 'UnitB7', 'UnitB8', 'UnitB9', 'UnitB10',
                 'UnitB11', 'UnitB12', 'UnitB13', 'UnitB14', 'UnitB15', 'UnitB16', 'UnitB17', 'UnitB18', 'UnitB19', 'UnitB20',
                 'UnitB21', 'UnitB22', 'UnitB23', 'UnitB24', 'UnitB25', 'UnitB26', 'UnitB27', 'UnitB28', 'UnitB29', 'UnitB30',
                 'UnitB31', 'UnitB32', 'UnitB33', 'UnitB34', 'UnitB35', 'UnitB36', 'UnitB37', 'UnitB38', 'UnitB39', 'UnitB40',
                 'UnitB41', 'UnitB42', 'UnitB43', 'UnitB44', 'UnitB45', 'UnitB46', 'UnitB47', 'UnitB48', 'UnitB49', 'UnitB50',
                 'UnitB51', 'UnitB52', 'UnitB53', 'UnitB54', 'UnitB55', 'UnitB56', 'UnitB57', 'UnitB58', 'UnitB59', 'UnitB60',
                 'UnitB61', 'UnitB62', 'UnitB63', 'UnitB64', 'UnitB65', 'UnitB66', 'UnitB67', 'UnitB68', 'UnitB69', 'UnitB70',
                 'UnitB71', 'UnitB72', 'UnitB73', 'UnitB74', 'UnitB75', 'UnitB76', 'UnitB77', 'UnitB78', 'UnitB79', 'UnitB80',
                 'UnitB81', 'UnitB82', 'UnitB83', 'UnitB84', 'UnitB85', 'UnitB86', 'UnitB87', 'UnitB88', 'UnitB89', 'UnitB90',
                 'UnitB91', 'UnitB92', 'UnitB93', 'UnitB94', 'UnitB95', 'UnitB96', 'UnitB97', 'UnitB98', 'UnitB99', 'UnitB100',
                 'UnitB101', 'UnitB102', 'UnitB103', 'UnitB104', 'UnitB105', 'UnitB106', 'UnitB107', 'UnitB108', 'UnitB109', 'UnitB110',
                 'UnitB111', 'UnitB112', 'UnitB113', 'UnitB114', 'UnitB115', 'UnitB116', 'UnitB117', 'UnitB118', 'UnitB119', 'UnitB120',
                 'UnitB121', 'UnitB122', 'UnitB123', 'UnitB124', 'UnitB125', 'UnitB126', 'UnitB127', 'UnitB128', 'UnitB129', 'UnitB130',
                 'UnitB131', 'UnitB132', 'UnitB133', 'UnitB134', 'UnitB135', 'UnitB136', 'UnitB137', 'UnitB138', 'UnitB139', 'UnitB140',
                 'UnitB141', 'UnitB142', 'UnitB143', 'UnitB144', 'UnitB145', 'UnitB146', 'UnitB147', 'UnitB148', 'UnitB149', 'UnitB150',
                 'UnitB151', 'UnitB152', 'UnitB153', 'UnitB154', 'UnitB155', 'UnitB156', 'UnitB157', 'UnitB158', 'UnitB159', 'UnitB160',
                 'UnitB161', 'UnitB162', 'UnitB163', 'UnitB164', 'UnitB165', 'UnitB166', 'UnitB167', 'UnitB168', 'UnitB169', 'UnitB170',
                 'UnitB171', 'UnitB172', 'UnitB173', 'UnitB174', 'UnitB175', 'UnitB176', 'UnitB177', 'UnitB178', 'UnitB179', 'UnitB180',
                 'UnitB181', 'UnitB182', 'UnitB183', 'UnitB184', 'UnitB185', 'UnitB186', 'UnitB187', 'UnitB188', 'UnitB189', 'UnitB190',
                 'UnitB191', 'UnitB192', 'UnitB193', 'UnitB194', 'UnitB195', 'UnitB196', 'UnitB197', 'UnitB198', 'UnitB199', 'UnitB200',
                 'UnitANext1', 'UnitANext2', 'UnitANext3', 'UnitANext4', 'UnitANext5', 'UnitANext6', 'UnitANext7', 'UnitANext8',
                 'UnitANext9', 'UnitANext10','UnitANext11', 'UnitANext12', 'UnitANext13', 'UnitANext14', 'UnitANext15', 'UnitANext16',
                 'UnitANext17', 'UnitANext18', 'UnitANext19', 'UnitANext20', 'UnitANext21', 'UnitANext22', 'UnitANext23', 'UnitANext24',
                 'UnitANext25', 'UnitANext26', 'UnitANext27', 'UnitANext28', 'UnitANext29', 'UnitANext30', 'UnitANext31', 'UnitANext32',
                 'UnitANext33', 'UnitANext34', 'UnitANext35', 'UnitANext36', 'UnitANext37', 'UnitANext38', 'UnitANext39', 'UnitANext40',
                 'UnitANext41', 'UnitANext42', 'UnitANext43', 'UnitANext44', 'UnitANext45', 'UnitANext46', 'UnitANext47', 'UnitANext48',
                 'UnitANext49', 'UnitANext50', 'UnitANext51', 'UnitANext52', 'UnitANext53', 'UnitANext54', 'UnitANext55', 'UnitANext56',
                 'UnitANext57', 'UnitANext58', 'UnitANext59', 'UnitANext60', 'UnitANext61', 'UnitANext62', 'UnitANext63', 'UnitANext64',
                 'UnitANext65', 'UnitANext66', 'UnitANext67', 'UnitANext68', 'UnitANext69', 'UnitANext70', 'UnitANext71', 'UnitANext72',
                 'UnitANext73', 'UnitANext74', 'UnitANext75', 'UnitANext76', 'UnitANext77', 'UnitANext78', 'UnitANext79', 'UnitANext80',
                 'UnitANext81', 'UnitANext82', 'UnitANext83', 'UnitANext84', 'UnitANext85', 'UnitANext86', 'UnitANext87', 'UnitANext88',
                 'UnitANext89', 'UnitANext90', 'UnitANext91', 'UnitANext92', 'UnitANext93', 'UnitANext94', 'UnitANext95', 'UnitANext96',
                 'UnitANext97', 'UnitANext98', 'UnitANext99', 'UnitANext100', 'UnitANext101', 'UnitANext102', 'UnitANext103', 'UnitANext104',
                 'UnitANext105', 'UnitANext106', 'UnitANext107', 'UnitANext108', 'UnitANext109', 'UnitANext110', 'UnitANext111', 'UnitANext112',
                 'UnitANext113', 'UnitANext114', 'UnitANext115', 'UnitANext116', 'UnitANext117', 'UnitANext118', 'UnitANext119', 'UnitANext120',
                 'UnitANext121', 'UnitANext122', 'UnitANext123', 'UnitANext124', 'UnitANext125', 'UnitANext126', 'UnitANext127', 'UnitANext128',
                 'UnitANext129', 'UnitANext130', 'UnitANext131', 'UnitANext132', 'UnitANext133', 'UnitANext134', 'UnitANext135', 'UnitANext136',
                 'UnitANext137', 'UnitANext138', 'UnitANext139', 'UnitANext140', 'UnitANext141', 'UnitANext142', 'UnitANext143', 'UnitANext144',
                 'UnitANext145', 'UnitANext146', 'UnitANext147', 'UnitANext148', 'UnitANext149', 'UnitANext150', 'UnitANext151', 'UnitANext152',
                 'UnitANext153', 'UnitANext154', 'UnitANext155', 'UnitANext156', 'UnitANext157', 'UnitANext158', 'UnitANext159', 'UnitANext160',
                 'UnitANext161', 'UnitANext162', 'UnitANext163', 'UnitANext164', 'UnitANext165', 'UnitANext166', 'UnitANext167', 'UnitANext168',
                 'UnitANext169', 'UnitANext170', 'UnitANext171', 'UnitANext172', 'UnitANext173', 'UnitANext174', 'UnitANext175', 'UnitANext176',
                 'UnitANext177', 'UnitANext178', 'UnitANext179', 'UnitANext180', 'UnitANext181', 'UnitANext182', 'UnitANext183', 'UnitANext184',
                 'UnitANext185', 'UnitANext186', 'UnitANext187', 'UnitANext188', 'UnitANext189', 'UnitANext190', 'UnitANext191', 'UnitANext192',
                 'UnitANext193', 'UnitANext194', 'UnitANext195', 'UnitANext196', 'UnitANext197', 'UnitANext198', 'UnitANext199', 'UnitANext200',
                 'UnitBNext1', 'UnitBNext2', 'UnitBNext3', 'UnitBNext4', 'UnitBNext5', 'UnitBNext6', 'UnitBNext7', 'UnitBNext8',
                 'UnitBNext9', 'UnitBNext10', 'UnitBNext11', 'UnitBNext12', 'UnitBNext13', 'UnitBNext14', 'UnitBNext15', 'UnitBNext16',
                 'UnitBNext17', 'UnitBNext18', 'UnitBNext19', 'UnitBNext20', 'UnitBNext21', 'UnitBNext22', 'UnitBNext23', 'UnitBNext24',
                 'UnitBNext25', 'UnitBNext26', 'UnitBNext27', 'UnitBNext28', 'UnitBNext29', 'UnitBNext30', 'UnitBNext31', 'UnitBNext32',
                 'UnitBNext33', 'UnitBNext34', 'UnitBNext35', 'UnitBNext36', 'UnitBNext37', 'UnitBNext38', 'UnitBNext39', 'UnitBNext40',
                 'UnitBNext41', 'UnitBNext42', 'UnitBNext43', 'UnitBNext44', 'UnitBNext45', 'UnitBNext46', 'UnitBNext47', 'UnitBNext48',
                 'UnitBNext49', 'UnitBNext50', 'UnitBNext51', 'UnitBNext52', 'UnitBNext53', 'UnitBNext54', 'UnitBNext55', 'UnitBNext56',
                 'UnitBNext57', 'UnitBNext58', 'UnitBNext59', 'UnitBNext60', 'UnitBNext61', 'UnitBNext62', 'UnitBNext63', 'UnitBNext64',
                 'UnitBNext65', 'UnitBNext66', 'UnitBNext67', 'UnitBNext68', 'UnitBNext69', 'UnitBNext70', 'UnitBNext71', 'UnitBNext72',
                 'UnitBNext73', 'UnitBNext74', 'UnitBNext75', 'UnitBNext76', 'UnitBNext77', 'UnitBNext78', 'UnitBNext79', 'UnitBNext80',
                 'UnitBNext81', 'UnitBNext82', 'UnitBNext83', 'UnitBNext84', 'UnitBNext85', 'UnitBNext86', 'UnitBNext87', 'UnitBNext88',
                 'UnitBNext89', 'UnitBNext90', 'UnitBNext91', 'UnitBNext92', 'UnitBNext93', 'UnitBNext94', 'UnitBNext95', 'UnitBNext96',
                 'UnitBNext97', 'UnitBNext98', 'UnitBNext99', 'UnitBNext100', 'UnitBNext101', 'UnitBNext102', 'UnitBNext103', 'UnitBNext104',
                 'UnitBNext105', 'UnitBNext106', 'UnitBNext107', 'UnitBNext108', 'UnitBNext109', 'UnitBNext110', 'UnitBNext111', 'UnitBNext112',
                 'UnitBNext113', 'UnitBNext114', 'UnitBNext115', 'UnitBNext116', 'UnitBNext117', 'UnitBNext118', 'UnitBNext119', 'UnitBNext120',
                 'UnitBNext121', 'UnitBNext122', 'UnitBNext123', 'UnitBNext124', 'UnitBNext125', 'UnitBNext126', 'UnitBNext127', 'UnitBNext128',
                 'UnitBNext129', 'UnitBNext130', 'UnitBNext131', 'UnitBNext132', 'UnitBNext133', 'UnitBNext134', 'UnitBNext135', 'UnitBNext136',
                 'UnitBNext137', 'UnitBNext138', 'UnitBNext139', 'UnitBNext140', 'UnitBNext141', 'UnitBNext142', 'UnitBNext143', 'UnitBNext144',
                 'UnitBNext145', 'UnitBNext146', 'UnitBNext147', 'UnitBNext148', 'UnitBNext149', 'UnitBNext150', 'UnitBNext151', 'UnitBNext152',
                 'UnitBNext153', 'UnitBNext154', 'UnitBNext155', 'UnitBNext156', 'UnitBNext157', 'UnitBNext158', 'UnitBNext159', 'UnitBNext160',
                 'UnitBNext161', 'UnitBNext162', 'UnitBNext163', 'UnitBNext164', 'UnitBNext165', 'UnitBNext166', 'UnitBNext167', 'UnitBNext168',
                 'UnitBNext169', 'UnitBNext170', 'UnitBNext171', 'UnitBNext172', 'UnitBNext173', 'UnitBNext174', 'UnitBNext175', 'UnitBNext176',
                 'UnitBNext177', 'UnitBNext178', 'UnitBNext179', 'UnitBNext180', 'UnitBNext181', 'UnitBNext182', 'UnitBNext183', 'UnitBNext184',
                 'UnitBNext185', 'UnitBNext186', 'UnitBNext187', 'UnitBNext188', 'UnitBNext189', 'UnitBNext190', 'UnitBNext191', 'UnitBNext192',
                 'UnitBNext193', 'UnitBNext194', 'UnitBNext195', 'UnitBNext196', 'UnitBNext197', 'UnitBNext198', 'UnitBNext199', 'UnitBNext200']

# preprocess datasets
for i in ds_list:
    print("Preparing %s dataset..." % i)

    pairs = read_source(ctdb_filepath = 'CDTB/zho.pdtb.cdtb_%s.tok' % i,
                        emb_filepath  = 'Embeddings/tencent_ailab_embedding_zh_d200_v0.2.0_s.txt')

    match_df = match(pairs, dfs[i])

    df = pd.DataFrame(match_df, columns = column_names)

    df.to_csv('Datasets/dataset_%s.csv' % i)

    print("Done!")