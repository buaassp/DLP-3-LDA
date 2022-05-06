import math
import jieba
import os 
import random
import numpy as np
##################################################################################################
##本程序参考了https://blog.csdn.net/shzx_55733/article/details/116280982?spm=1001.2014.3001.5502文章
##################################################################################################


def deta_deal(content):  #处理语料库，后面会用到，在第一次作业时已经使用过
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        content = content.replace(a, '')
    return content



def read_novel(path):  # 读取小说内容，主要是按照题目要求读取段落
    cont = []
    names = os.listdir(path)
    for name in names:
            novel_name = path + '\\' + name
            con_text = []
            with open(novel_name, 'r', encoding='ANSI') as data:
                cont_data = data.read()
                cont_data = deta_deal(cont_data)     #处理数据
                cont_data = jieba.lcut(cont_data)    #结巴分词
                con_list = list(cont_data)           
                p = int(len(cont_data)//12          )#16篇文章均匀选出200段，可以先每篇文章均匀选出12段
               #因为12*16=192，少了8段，为了保证总字数不变，将每一段的字数调整为521字，保证均匀
                for i in range(13):
                    con_text = con_text + con_list[i*p:i*p+520]
                cont.append(con_text)
            data.close()
    return cont, names
def read_novel_text(path):  # 读取小说内容，主要是按照题目要求读取段落
    cont = []
    names = os.listdir(path)
    for name in names:
            novel_name = path + '\\' + name
            con_text = []
            with open(novel_name, 'r', encoding='ANSI') as data:
                cont_data = data.read()
                cont_data = deta_deal(cont_data)     #处理数据
                cont_data = jieba.lcut(cont_data)    #结巴分词
                con_list = list(cont_data)           
                p = int(len(cont_data)//12          )#16篇文章均匀选出200段，可以先每篇文章均匀选出12段
               #因为12*16=192，少了8段，为了保证总字数不变，将每一段的字数调整为521字，保证均匀
                for i in range(13):
                    con_text = con_text + con_list[i*p+521:i*p+1041]
                cont.append(con_text)
            data.close()
    return cont, names



if __name__ == '__main__':

    ##########这里开始是对模型的训练#########
    ##初始化模型
    [data_txt, files] = read_novel("dataset")   #按照段落要求读取小说内容
    print(files)
    Topic_EVE = []  # 统计每个词来自什么主题，也就是每个词来自哪个文章
    Topic_count = {}  # 每个文章下有多少词
    Topic_fre0 = {}; Topic_fre1 = {}; Topic_fre2 = {}; Topic_fre3 = {}
    Topic_fre4 = {}; Topic_fre5 = {}; Topic_fre6 = {}; Topic_fre7 = {}
    Topic_fre8 = {}; Topic_fre9 = {}; Topic_fre10 = {}; Topic_fre11 = {}
    Topic_fre12 = {}; Topic_fre13 = {}; Topic_fre14 = {}; Topic_fre15 = {}  #储存16篇文章的词频

    word_count = []  # 每篇文章中有多少个词
    word_fre = []    # 每篇文章主题词的词频
    i = 0
    for data in data_txt:
        topic = []
        docfre = {}
        for word in data:
            a = random.randint(0, len(data_txt)-1)          # 每个词赋予一个随机初始主题
            topic.append(a)
            if '\u4e00' <= word <= '\u9fa5':
                Topic_count[a] = Topic_count.get(a, 0) + 1  # 计算每主题总词数
                docfre[a] = docfre.get(a, 0) + 1            # 计算每篇文章主题词的词频
                exec('Topic_fre{}[word]=Topic_fre{}.get(word, 0) + 1'.format(i, i))  # 计算每个的词频
        Topic_EVE.append(topic)
        docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
        word_fre.append(docfre)
        word_count.append(sum(docfre))  # 统计每篇文章的总词数
        i += 1
    Topic_count = list(dict(sorted(Topic_count.items(), key=lambda x: x[0], reverse=False)).values())
    word_fre = np.array(word_fre)  
    Topic_count = np.array(Topic_count)  
    Doc_count = np.array(word_count)  

    Word_eve = []        # 选中每个主题的概率
    Word_eve1 = []       # 迭代后选中每个主题概率

    for i in range(len(data_txt)):
        doc = np.divide(word_fre[i], Doc_count[i])
        Word_eve.append(doc)
    Word_eve = np.array(Word_eve)
    
    stop = 0  # 迭代停止标志
    loop_count = 1  # 迭代次数
    while stop == 0:
        i = 0
        for data in data_txt:
            top = Topic_EVE[i]
            for w in range(len(data)):
                word = data[w]
                pro = []
                topfre = []
                if '\u4e00' <= word <= '\u9fa5':
                    for j in range(len(data_txt)):
                        exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))  # 计算每个的词频
                    pro = Word_eve[i] * topfre / Topic_count  # 得到该词出现的概率向量
                    m = np.argmax(pro)                       # 认为该词是由上述概率之积最大的那个topic产生的
                    word_fre[i][top[w]] -= 1                 # 更新每个文档有多少各个主题的词
                    word_fre[i][m] += 1
                    Topic_count[top[w]] -= 1                 # 更新每个主题的总词数
                    Topic_count[m] += 1
                    exec('Topic_fre{}[word] = Topic_fre{}.get(word, 0) - 1'.format(top[w], top[w]))  # 更新每个的词频
                    exec('Topic_fre{}[word] = Topic_fre{}.get(word, 0) + 1'.format(m, m))
                    top[w] = m
            Topic_EVE[i] = top
            i += 1

        if loop_count == 1:  # 计算每篇文章主题概率
            for i in range(len(data_txt)):
                doc = np.divide(word_fre[i], Doc_count[i])
                Word_eve1.append(doc)
            Word_eve1 = np.array(Word_eve1)
        else:
            for i in range(len(data_txt)):
                doc = np.divide(word_fre[i], Doc_count[i])
                Word_eve1[i] = doc
       # print('训练前主题概率为：',Word_eve)
       # print('每次迭代后主题概率为',(Word_eve1))
        if (Word_eve1 == Word_eve).all():  
            stop = 1
        else:
            Word_eve = Word_eve1.copy()
        loop_count += 1

    print('最终训练结果为：',Word_eve1)  # 输出训练后选中每个主题概率
    print('迭代次数为：',loop_count)  # 输出迭代次数
    print('Training is complete')

#########这里开始对训练好的模型做测试##############
    [test_txt, files] = read_novel_text("dataset")
    Doc_count_test = []     # 文章总词数
    Doc_fre_test = []       # 文章中主题出现频率
    Topic_All_test = []     # 主题中关键词频率
    i = 0
    for data in test_txt:
        topic = []
        docfre = {}
        for word in data:
            a = random.randint(0, len(data_txt) - 1)  # 测试，为每一个词赋予一个随机主题
            topic.append(a)
            if '\u4e00' <= word <= '\u9fa5':
                docfre[a] = docfre.get(a, 0) + 1  # 计算词频
        Topic_All_test.append(topic)
        docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
        Doc_fre_test.append(docfre)
        Doc_count_test.append(sum(docfre))  # 计算每篇文章总次数，并存储
        i += 1
 
    Doc_fre_test = np.array(Doc_fre_test)
    Doc_count_test = np.array(Doc_count_test)
    print('测试文章主题出现概率',Doc_fre_test)
    print('每篇文章出现总次数',Doc_count_test)
    Doc_pro_test = []       # 与上面相似，计算每个主题被选中的概率
    Doc_pronew_test = []    # 与上一步类似，计算迭代后每个主题被选中的概率
    for i in range(len(test_txt)):
        doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
        Doc_pro_test.append(doc)
    Doc_pro_test = np.array(Doc_pro_test)
    print('每个主题被选中概率',Doc_pro_test)
    stop = 0            # 迭代停止标志
    loop_count = 1      # 迭代次数
    while stop == 0:
        i = 0
        for data in test_txt:
            top = Topic_All_test[i]
            for w in range(len(data)):
                word = data[w]
                pro = []
                topfre = []
                if '\u4e00' <= word <= '\u9fa5':
                    for j in range(len(data_txt)):
                        exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))  # 计算词频
                    pro = Doc_pro_test[i] * topfre / Topic_count  # 得到概率向量
                    m = np.argmax(pro)  
                    Doc_fre_test[i][top[w]] -= 1  # 计算文章中主题词的个数
                    Doc_fre_test[i][m] += 1
                    top[w] = m
            Topic_All_test[i] = top
            i += 1
        if loop_count == 1:  # 计算新的每篇文章选中主题的概率
            for i in range(len(test_txt)):
                doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
                Doc_pronew_test.append(doc)
            Doc_pronew_test = np.array(Doc_pronew_test)
        else:
            for i in range(len(test_txt)):
                doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
                Doc_pronew_test[i] = doc
       # print('每个主题被选中概率',Doc_pro_test)
       # print('迭代后每个主题被选中概率',Doc_pronew_test)
        if (Doc_pronew_test == Doc_pro_test).all():  # 主题概率不变 认为迭代结束
            stop = 1
        else:
            Doc_pro_test = Doc_pronew_test.copy()
        loop_count += 1

    print('测试集迭代后主题被选中概率：',Doc_pronew_test)
    print('迭代次数',loop_count)
    print('Test is complete')
    ##开始表示计算结果
    result = []
    for k in range(len(test_txt)):
        pro = []
        for i in range(len(data_txt)):
            d = 0
            for j in range(len(data_txt)):
                d += (Word_eve[i][j] - Doc_pro_test[k][j])**2  # 计算欧式距离。欧式距离小的认为对应文章主题
            pro.append(d)
        m = pro.index(min(pro))
        print('输出欧式距离为：',pro)
        result.append(m)
    print('读入的文件名：',files)
    print('输出结果为：',result)

