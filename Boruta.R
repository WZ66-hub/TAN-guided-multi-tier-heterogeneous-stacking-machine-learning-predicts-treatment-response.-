
library(readr)
library(Boruta)
library(ggplot2)
## 设置随机种子保证结果的可重复性
set.seed(1234)
dev <- read_csv("")

Var.Selec <- Boruta(group~.,   #构建公式
                    
                    data=dev,  #特征筛选所用数据集
                    
                    pValue = 0.05,  #置信水平，默认值0.01
                    
                    mcAdj = TRUE,  #使用Bonferroni方法的多重比较调整
                    
                    maxRuns = 500,   #最大迭代次数
                    
                    doTrace = 0,  #运行报告冗长等级
                    
                    holdHistory = TRUE,   #储存重要历史记录
                    
                    getImp = getImpRfZ)   #获取属性重要性


print(Var.Selec)

# 设置图形输出设备
png("Boruta_ImpHistory.png", width = 1200, height = 800, res = 200)
# 绘制重要性历史图
Boruta::plotImpHistory(Var.Selec) +
  theme(text = element_text(size = 14),  # 增大字体大小
        axis.text = element_text(size = 12),
        legend.text = element_text(size = 12),
        plot.title = element_text(size = 16, face = "bold"))

dev.off()
