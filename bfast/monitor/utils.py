from datetime import datetime

import statsmodels.api as sm
from scipy import stats, optimize
import numpy as np
import pandas

__critvals = np.array([1.22762665817831, 1.68732328328598, 2.22408818231127,
1.33623105388957, 1.88633090100410, 2.70443676308234, 1.34108685187662,
1.89958445059510, 2.73714807589866, 1.34165681503561, 1.90129850983572,
2.74287924359951, 1.34182451007628, 1.90200317899371, 2.74592761324742,
1.23066970866923, 1.69160084179475, 2.23167217815961, 1.33879059030532,
1.89023763912529, 2.71316397536365, 1.34391583752137, 1.90372282728502,
2.74320513885208, 1.34413752134022, 1.90516570855328, 2.74972282830356,
1.34439131451376, 1.90575941243566, 2.75332577427708, 1.23276485917986,
1.69633389358351, 2.23855562552750, 1.34146823757119, 1.89505055324671,
2.72230751827108, 1.34624172867029, 1.90786349815282, 2.75022696456136,
1.34645624006547, 1.90937175063387, 2.75749248513154, 1.34660319011975,
1.91003224799722, 2.76033090809588, 1.23564055536802, 1.70158435524857,
2.24671622929800, 1.34417861992375, 1.89968720697260, 2.73013628548337,
1.34839917683048, 1.91210039341598, 2.75779534589076, 1.34885191571363,
1.91371639502306, 2.76545067687381, 1.34915116589537, 1.91430116495002,
2.76795746388208, 1.23847794049457, 1.70551689505972, 2.25495483554988,
1.34658597857070, 1.90396104830582, 2.73783443325526, 1.35109583245859,
1.91648497862793, 2.76609400488216, 1.35155402000607, 1.91794056113743,
2.77239828238273, 1.35178587051693, 1.91852119221045, 2.77449341898917,
1.24198076615651, 1.71107257285131, 2.26367247535366, 1.34920669388068,
1.90830710813031, 2.74528962424803, 1.35376511356222, 1.92091285840255,
2.77292540014255, 1.35398617621481, 1.92232123912668, 2.78065638359198,
1.35417886044462, 1.92363924258536, 2.78377150533344, 1.24481637165883,
1.71626606143193, 2.27198147883231, 1.35202040971813, 1.91295018738968,
2.7535013308307, 1.35614865096182, 1.92625368162383, 2.78258094839249,
1.35634491518239, 1.92759911760505, 2.78832963965467, 1.3566836167395,
1.92813013037837, 2.79040905895933, 1.24879019056867, 1.72019347712026,
2.28029374580499, 1.35462026215463, 1.91751604106457, 2.76165479244638,
1.35891530342030, 1.93110016008493, 2.78995735838407, 1.35923481408015,
1.93247588823320, 2.79587781452729, 1.35948739899674, 1.93318354531244,
2.79791269784148, 1.25239505325696, 1.72583651612689, 2.28997972044219,
1.35719673301301, 1.92212903036675, 2.76995006883156, 1.36194735830278,
1.93617083100785, 2.79796564416274, 1.36226456490384, 1.93770964117028,
2.80461288523030, 1.36256898537193, 1.93819181548142, 2.80812472032002,
1.25498907533676, 1.73080136283287, 2.30139150759728, 1.36049997713998,
1.92796698329961, 2.77784350063502, 1.36523574166444, 1.94187011880212,
2.80837367523907, 1.36560026937241, 1.94330305432000, 2.81331080938179,
1.36577231193638, 1.94372394322678, 2.81585876288918, 1.25822933229468,
1.73632152412507, 2.31039779234819, 1.36372549810846, 1.93324324569141,
2.78801292378179, 1.36827681119024, 1.9472427855027, 2.81622523964660,
1.36850460298850, 1.94888292783326, 2.82132245132351, 1.36886320957607,
1.94920727427515, 2.82427033701458, 1.26223904797321, 1.74196445041873,
2.32031649132490, 1.36667883846684, 1.93918038241824, 2.79690003525357,
1.37164275186352, 1.95211092153476, 2.82544776912852, 1.37214926058743,
1.95353915653814, 2.83191307845068, 1.37237427017584, 1.95410938212938,
2.8345078182179, 1.26596555529583, 1.74739377676382, 2.32921900751217,
1.37052411847341, 1.945610384202, 2.80782434173153, 1.37460434441364,
1.95776813619084, 2.83581742506166, 1.37477725947058, 1.95924434085755,
2.84091691021233, 1.37485167515552, 1.95942574616048, 2.84343425522273,
1.26923998208695, 1.75350003177274, 2.33930632358443, 1.37374212322454,
1.95099093303112, 2.81710131124371, 1.37781526603652, 1.96349092943121,
2.84607795173187, 1.37813432738452, 1.96487051905954, 2.85104635740391,
1.37831535555367, 1.96506931590280, 2.85360428459958, 1.27264091504165,
1.75880466474247, 2.35063208445546, 1.37657697158592, 1.95723350782807,
2.82810279142842, 1.38115250250291, 1.96961608411167, 2.85653311722288,
1.38154788481536, 1.97057577644158, 2.86061643301292, 1.38175120511989,
1.97097369401268, 2.86243265700872, 1.27603992297831, 1.76547247515605,
2.36239255307403, 1.38022128723907, 1.96320972852135, 2.83902929955712,
1.38474992975096, 1.97448554973337, 2.86615061088975, 1.38507423359833,
1.97560947773326, 2.87105813546864, 1.38537793038039, 1.97593043984907,
2.87265383703287, 1.27959232745831, 1.77251787804276, 2.37368186178517,
1.38394338302155, 1.9696910872405, 2.84985602895211, 1.38801205099788,
1.98048061846908, 2.87507755322594, 1.38836794783610, 1.98147802794651,
2.87864303912487, 1.38847324240149, 1.98160704981630, 2.88094199122226,
1.28485863206623, 1.77940231593083, 2.38399333407731, 1.38769333359217,
1.97536757884356, 2.85943741078032, 1.39091294470053, 1.98572396037020,
2.88546120078802, 1.39117348693852, 1.98700655564388, 2.88982113905153,
1.39145597087778, 1.98735507741592, 2.89140196762897, 1.28918350549942,
1.78802901378588, 2.39655947720336, 1.39048392473728, 1.98148833656383,
2.87206236687600, 1.39464610643847, 1.99228524462093, 2.89647860951724,
1.39508146109301, 1.99311591109480, 2.90019387355943, 1.39532955755002,
1.99344303519556, 2.90133608008974, 1.29311889830519, 1.79524119452131,
2.40900289159466, 1.39408189315686, 1.98760402287577, 2.88195698007885,
1.39872249131114, 1.99798543677965, 2.90646370076569, 1.39885993994357,
1.99891607446647, 2.91100628450779, 1.39911228012838, 1.99907859690203,
2.91248740560158, 1.29774316107240, 1.8022227519734, 2.42051303184748,
1.39843637731271, 1.99470655040184, 2.89422935489322, 1.40265414814487,
2.00481875821914, 2.91830366891529, 1.40307434720965, 2.00650282065195,
2.92088700191003, 1.40318773259415, 2.00698531362174, 2.92234001556265,
1.30293027348104, 1.80915784114738, 2.43187800124754, 1.40284080982133,
2.0014831952455, 2.90588801177386, 1.40676218000241, 2.01232512752593,
2.92793049243892, 1.40727611318749, 2.01325663610310, 2.93136693477480,
1.40748954375846, 2.01348511642423, 2.93310201832971, 1.30709536503133,
1.81682673045196, 2.44213404011556, 1.40734331669295, 2.00990617784639,
2.91830314114826, 1.41145745016112, 2.01966763484381, 2.93994000374386,
1.41163904123871, 2.02055082936700, 2.94237254586805, 1.41181372331003,
2.02095905934155, 2.94366180788666, 1.31158624473196, 1.82485724271212,
2.45546454151652, 1.41189572366036, 2.01800884070615, 2.92889241096725,
1.41537372778588, 2.02771381041308, 2.9513361640116, 1.41558214392260,
2.02869455951327, 2.95513502136229, 1.41569840195645, 2.02936696517065,
2.95594174585483, 1.31737646083926, 1.83334265228127, 2.46823974105248,
1.41597082120943, 2.02639046609307, 2.94165001703407, 1.41936982775548,
2.03511516275574, 2.96290806233328, 1.41956165048630, 2.03595001884676,
2.96514551259539, 1.41977723105019, 2.03644766225911, 2.96689759268915,
1.32335166269351, 1.84186416555787, 2.48305431089417, 1.42022026752427,
2.03402198587187, 2.95537968710744, 1.42362459481656, 2.04266224216936,
2.97653791054636, 1.42380433491698, 2.04423035024309, 2.97934036146678,
1.42381861952310, 2.04438785665452, 2.98001396428591, 1.32786358742455,
1.85177121092134, 2.50014264198274, 1.42524057175581, 2.04207854563661,
2.96828782383848, 1.42868230819686, 2.05212705707684, 2.98956874067662,
1.42884928919173, 2.05317292848793, 2.99268186004286, 1.42895669363032,
2.05338145412747, 2.99480828166113, 1.33350696717069, 1.8612108588202,
2.51277454011338, 1.43085920948697, 2.05201428309035, 2.98345815507671,
1.43352636120560, 2.05842768327946, 3.00281714459409, 1.43362752470632,
2.05914487796479, 3.00714447420536, 1.43363936425424, 2.05937675809378,
3.00867661799182, 1.33919230324655, 1.86950504404675, 2.52729044406123,
1.43555358483841, 2.05894590307482, 3.00088564120662, 1.43825178128014,
2.06555482615441, 3.01767474219581, 1.43839225150709, 2.06609364605974,
3.02151497458958, 1.43840514935644, 2.06616171881744, 3.02246324396029,
1.34492551588986, 1.87858666618536, 2.54412126382912, 1.44053121853627,
2.06609460804903, 3.01450613235098, 1.44291380981225, 2.07404845300542,
3.03164032383581, 1.44297415629377, 2.07470796502465, 3.03325532373884,
1.44307576181773, 2.07473789571604, 3.03394218534334, 1.35027274948328,
1.88695252785603, 2.55992945172055, 1.44543973881029, 2.07553346343271,
3.03021741060986, 1.44812318488765, 2.08244751617481, 3.04573310198876,
1.44822824145847, 2.0828482466276, 3.04844221000558, 1.44823608804569,
2.08287027642156, 3.04928862144923, 1.35619676871919, 1.89927100396680,
2.57990545930011, 1.45089202071443, 2.08564859894760, 3.04514305197666,
1.45314333050063, 2.09215841827476, 3.06270932788493, 1.45326230316142,
2.09253295606338, 3.06521651133586, 1.45331084387002, 2.09256864345591,
3.06559847575043, 1.36259049396895, 1.90998994145834, 2.59954495128891,
1.45602813899372, 2.09526908882824, 3.06318938472853, 1.45889808672432,
2.10175646231178, 3.08105683136056, 1.45899173457736, 2.10192808009171,
3.08396500595724, 1.45902908335449, 2.10195202161131, 3.08538712566800,
1.36957894266347, 1.92044406545023, 2.62226086016471, 1.46278181190807,
2.10508945592555, 3.08204384920415, 1.46547805028961, 2.11155420066774,
3.09911181554597, 1.46554748937219, 2.11163306195199, 3.10276276238258,
1.46557816998587, 2.11178013692067, 3.10344121252328, 1.37645783098199,
1.93325881936299, 2.64328186531920, 1.46985441198876, 2.11499858552887,
3.10118717953729, 1.47234905574779, 2.12143821941805, 3.11864517719780,
1.47245299778231, 2.12152134245074, 3.12128659480993, 1.47253053865670,
2.12170170901203, 3.12169013198376, 1.38428233380854, 1.94846412205370,
2.66757121215124, 1.47696589701722, 2.12644605226194, 3.12137345421518,
1.48040430543862, 2.13384118696042, 3.14357440726387, 1.48055874421042,
2.13393551264924, 3.14504206811941, 1.48057636069157, 2.13419420963212,
3.14546814874483, 1.39194501572598, 1.96135719471953, 2.68912688266154,
1.48523183187464, 2.13860160003387, 3.14629361944769, 1.48742657759599,
2.14418189402044, 3.16175551098466, 1.48752188622688, 2.14420431912788,
3.1637257017909, 1.48759284416558, 2.14425304938479, 3.16409611837069,
1.40100798604519, 1.97830731677393, 2.71615273475211, 1.49283351395210,
2.15176436484615, 3.16975412064978, 1.49494270557386, 2.15661675767253,
3.18896143251092, 1.49506819639025, 2.15693361041496, 3.1920229659298,
1.49517101820054, 2.15761471934841, 3.19331642253382, 1.41166991783875,
1.99334933362960, 2.74499929176667, 1.50095115109795, 2.16599721679536,
3.19871150134074, 1.50325247631427, 2.17327200492716, 3.21409553789802,
1.50344238684623, 2.17354493026062, 3.21667996181354, 1.50384229730937,
2.17377124717081, 3.21712193181301, 1.42087734975763, 2.01053594213232,
2.76956419117839, 1.51067508324004, 2.18452237446875, 3.22557804826641,
1.51196972918776, 2.19079771530290, 3.23793630392874, 1.51198649641834,
2.19114045202173, 3.24004966347741, 1.51208390469150, 2.19161080737738,
3.24079319998886, 1.43326294742430, 2.03146339358619, 2.79961591720564,
1.51983679278537, 2.20116995886405, 3.2528296362616, 1.52159988186384,
2.20853516508261, 3.27400632880909, 1.52162850629023, 2.20875377365514,
3.27485973776243, 1.52164497279622, 2.20907282819197, 3.27693245691716,
1.44359950240666, 2.05289581866667, 2.84219702965024, 1.53269740247647,
2.21810904347873, 3.29448456006627, 1.53383764478810, 2.2243689090344,
3.30948160598604, 1.53408167478549, 2.22491132803164, 3.31063124079675,
1.53436454805016, 2.22538374189696, 3.31144207023956, 1.45511231203346,
2.07310241616824, 2.88262759124401, 1.54440348055754, 2.24107971112342,
3.32824508683748, 1.54538029292232, 2.24653424911614, 3.33991237704364,
1.54554689692936, 2.24671042441639, 3.34092246893143, 1.54556159000357,
2.24728555022134, 3.34121663787019, 1.47160701884111, 2.09696385947528,
2.92385031781064, 1.55910620044073, 2.26412295727692, 3.36800851346221,
1.56033776761057, 2.26914429720656, 3.38231671523284, 1.56033776761057,
2.26948708821640, 3.38333599579249, 1.56036062897291, 2.26978232260259,
3.38431258247793, 1.48885977022345, 2.12151951630359, 2.97162399662873,
1.57572071796450, 2.29053194168806, 3.41346594519685, 1.57658154293077,
2.29470772538911, 3.42383820313033, 1.57658154293077, 2.29516159555825,
3.42496818222524, 1.57673206447201, 2.29570318746438, 3.42513870373182,
1.50729964467061, 2.15191486007068, 3.02945819897377, 1.59695604364839,
2.32051977467861, 3.46025134389695, 1.59797111384019, 2.32525489078505,
3.47339264373285, 1.59797111384019, 2.32552183073452, 3.47422654487876,
1.59797111384019, 2.32552183073452, 3.47422654487876, 1.53175586508601,
2.20039670238309, 3.08704227453562, 1.61812223040947, 2.35590430637147,
3.51542896077976, 1.61839672202083, 2.35818246901503, 3.52934179098599,
1.61839672202083, 2.35917428857408, 3.5293634919398, 1.61839672202083,
2.35917428857408, 3.5293634919398, 1.56038054799786, 2.24711394446681,
3.17273589041428, 1.64940524297012, 2.40898177256112, 3.60624261804716,
1.64940524297012, 2.41186736779604, 3.61938625967275, 1.64940524297012,
2.41186736779604, 3.62048050863877, 1.64940524297012, 2.41186736779604,
3.62095881900556, 1.60458840049086, 2.30800393871034, 3.28942508259488,
1.68594331412298, 2.46453707538155, 3.71816408123168, 1.68594331412298,
2.46579678405964, 3.73434766642562, 1.68594331412298, 2.46579678405964,
3.73692000120501, 1.68594331412298, 2.46579678405964, 3.73697935597481,
1.67397676536158, 2.43457585062277, 3.45472681303725, 1.74550948894458,
2.56886151964252, 3.93535696186154, 1.74550948894458, 2.57025529149703,
3.94102916093653, 1.74550948894458, 2.57025529149703, 3.94102916093653,
1.74550948894458, 2.57025529149703, 3.94102916093653, 1.81954446481909,
2.22591384652163, 2.46391278744698, 2.08412863196499, 2.68085534937008,
3.33072912856251, 2.11180746892218, 2.74045835623640, 3.45640727356551,
2.12023606594727, 2.76014777604291, 3.50502362427164, 2.12320823869064,
2.76948388986443, 3.53381534472487, 1.82344798626059, 2.230802774778,
2.47126755950339, 2.08700050605196, 2.68643863041128, 3.33758996841233,
2.11463521114668, 2.74526328117409, 3.46306954978012, 2.12271312238526,
2.76418733961071, 3.51174962415423, 2.12592798765015, 2.77299763923263,
3.53986469765005, 1.82661414686887, 2.23546239454496, 2.47789609809196,
2.09015587299808, 2.6908988918078, 3.34574628680461, 2.11795938985215,
2.75000903381165, 3.46973908754325, 2.12570983180643, 2.76815169091552,
3.51823129600941, 2.12893373691857, 2.77792434665079, 3.54625771772862,
1.83012286026358, 2.24093729493409, 2.48559943269060, 2.09314772413788,
2.69650886566693, 3.35164057527594, 2.12094775157620, 2.75490767733301,
3.47590947015454, 2.12870146970024, 2.77167598773252, 3.52500056785711,
2.13144740982465, 2.7826132597037, 3.55264927591488, 1.83360988469104,
2.24691132720936, 2.49370946814411, 2.0962118341186, 2.70257736700686,
3.35975732382747, 2.12367749118468, 2.76002132079635, 3.4838819577291,
2.13138278881771, 2.77655954928124, 3.53096083903982, 2.13436228911229,
2.78701514364425, 3.55885056988873, 1.83715307775979, 2.25207640253111,
2.50169702442543, 2.09917174136122, 2.70764593946809, 3.36698550013957,
2.12700564954345, 2.76440130873053, 3.49160429317181, 2.13432755770033,
2.78146097026088, 3.53763539897956, 2.13712065817205, 2.79112022029345,
3.56535310539566, 1.84056756023769, 2.25711848954445, 2.51084263695293,
2.10278041385553, 2.71175246283891, 3.37452635531217, 2.13022885738685,
2.76901022857535, 3.49766509941435, 2.13721733095770, 2.78631355742898,
3.54494855968431, 2.13992421644020, 2.79577904040062, 3.57069206044516,
1.84447094408104, 2.26257788410066, 2.51772642451113, 2.10664115359282,
2.71747293878633, 3.38044996288511, 2.13331413213671, 2.77309254859561,
3.50516067930930, 2.14015064852228, 2.79087061339873, 3.55144998096019,
2.14299416667874, 2.80033862195106, 3.57741821964149, 1.84879237475351,
2.26937879553809, 2.52686751800841, 2.11051021733345, 2.72158437361444,
3.38753470587925, 2.13638901650338, 2.77940949186345, 3.5131671363738,
2.14337274683089, 2.79538164039360, 3.55844801303687, 2.14639323126965,
2.80542656199191, 3.58464718016958, 1.85298319721727, 2.27632863034704,
2.53508560150775, 2.11386339871615, 2.72700672699383, 3.39580947144644,
2.13957142023227, 2.78498660911762, 3.52077620774917, 2.14667091866216,
2.80000445263785, 3.56546708937295, 2.14981626879143, 2.81045609217290,
3.5925337751768, 1.85760168211944, 2.28251384825922, 2.54523365481574,
2.11800614299258, 2.73143146695321, 3.40358327963137, 2.14278319617785,
2.78984130595265, 3.52768748709407, 2.15023732394521, 2.80525351999000,
3.57154342050093, 2.15289921726444, 2.81495535248919, 3.59906345718385,
1.86229547366417, 2.28894633683292, 2.55414141834483, 2.12153141721421,
2.73662193058728, 3.41177944315769, 2.14654037225162, 2.79429396217219,
3.53425243724035, 2.15330108915549, 2.81091249083076, 3.57920520221927,
2.15654036991459, 2.82015982068815, 3.60516901034419, 1.86607380226072,
2.29685201489767, 2.56274334764907, 2.12493922632193, 2.74255386879242,
3.42016576209468, 2.15025122444992, 2.79938476001166, 3.5423682666881,
2.15708559803071, 2.81568046806229, 3.5869781137461, 2.15983848157569,
2.82517243302741, 3.61189624837151, 1.87015100669539, 2.30389287635647,
2.57155949666418, 2.12894072338200, 2.74847912565621, 3.42949501582605,
2.15350620192262, 2.80552173719103, 3.54921943409808, 2.16079879827035,
2.82118924879503, 3.59518899801846, 2.16354643287978, 2.8308412622816,
3.61829137565987, 1.87434883360663, 2.31063580959909, 2.58116077352148,
2.13256639487202, 2.75486166036281, 3.43823862826649, 2.15790353284222,
2.81162363606233, 3.556952953408, 2.16479117991533, 2.82650893062054,
3.60199425758701, 2.16725752708834, 2.83639345481454, 3.62622821191613,
1.87862404267598, 2.31859472417178, 2.59032031413285, 2.13624534724997,
2.76108741679843, 3.44841002477554, 2.16174960354069, 2.81625314022632,
3.56516503882035, 2.16837711066996, 2.83352040487738, 3.60850469838616,
2.17091203976378, 2.84230287019126, 3.63382432755621, 1.88404478043788,
2.32501638954249, 2.59843443819996, 2.13987494438998, 2.76740716652912,
3.45809823377464, 2.16591902110115, 2.82239472432692, 3.57347180354981,
2.17272178924074, 2.83862963483524, 3.61537720480582, 2.17575002492038,
2.84840763300983, 3.6399216842664, 1.88874302055418, 2.33313304952541,
2.60836613703340, 2.14428962586375, 2.77281938161961, 3.46817355099174,
2.16962692628423, 2.82852361821702, 3.58252656001226, 2.17696276396447,
2.84569765572058, 3.62443760121814, 2.17949319057116, 2.85427720899956,
3.64855929411662, 1.89345950619256, 2.34154552719839, 2.61986928303432,
2.1488466247166, 2.78146634553457, 3.47605925839772, 2.17482006047446,
2.83588674056336, 3.59178947066794, 2.1805518769006, 2.85153844900335,
3.63380685797965, 2.18278067782414, 2.86007218626277, 3.65706270350488,
1.8990209037878, 2.34991415430168, 2.63050978195371, 2.15289915148653,
2.78919143813695, 3.48805443257122, 2.17837241804329, 2.84236578638002,
3.60080824803134, 2.18427980386643, 2.85708296541044, 3.64032646794286,
2.18658263069226, 2.86587077200269, 3.66715043519786, 1.90449669183086,
2.35768939575221, 2.64203524567493, 2.15772291958247, 2.79537030239242,
3.49820780120991, 2.1823519062029, 2.84889368703037, 3.6085046209279,
2.18816464030844, 2.86285717062948, 3.65022038386188, 2.190831628588,
2.87216039748328, 3.67929490093814, 1.91062464756646, 2.36697288397517,
2.65184061301174, 2.16275539625543, 2.80181980497330, 3.50949974665316,
2.18644278886306, 2.85554284587873, 3.617755939703, 2.19270005873997,
2.87000840600416, 3.66011175030500, 2.19529666017785, 2.87833139479387,
3.68762130549175, 1.91580988964948, 2.37513621128066, 2.66228505736591,
2.16754418014273, 2.81000737272204, 3.52156441650787, 2.19144615710786,
2.8623506962233, 3.62735693507751, 2.19756475040665, 2.87641710595494,
3.67186260083532, 2.19953592798011, 2.88563012415683, 3.69607530343335,
1.92155884964439, 2.38436361932194, 2.67463123025473, 2.17340107059304,
2.81628043084800, 3.53204708984690, 2.19619036546802, 2.87000837096465,
3.63804587000029, 2.20137135641134, 2.88420384308174, 3.6839663778893,
2.20420219844996, 2.89167193279737, 3.70493839788555, 1.92769261345449,
2.39357722791474, 2.68841958585637, 2.17804833913143, 2.82471539156299,
3.54431829330582, 2.20084587599274, 2.87687892185265, 3.64855832140359,
2.20646049986697, 2.89077617284757, 3.69196098017429, 2.20976171342769,
2.89857506455057, 3.71711166453008, 1.93415193190404, 2.40318935014295,
2.70066425268548, 2.18292605491209, 2.83367986040498, 3.55624329579949,
2.20594262058077, 2.88476905540689, 3.65994343278081, 2.21235424451003,
2.89783163216231, 3.70252522625842, 2.21435123608395, 2.90758394761588,
3.72983287425351, 1.94041179292859, 2.41413112282927, 2.71215670623997,
2.18834249521883, 2.84211298312772, 3.56779426092527, 2.21151932219498,
2.8914629666975, 3.67385398834237, 2.21725202305910, 2.90711675054827,
3.71590528460708, 2.2197417573013, 2.9151394480235, 3.7410973195177,
1.94569298705534, 2.42440693533190, 2.72562580843461, 2.19418031959385,
2.85132825760796, 3.58056432859899, 2.21677557521056, 2.89960930083652,
3.68571416063101, 2.22390448303944, 2.91511703592227, 3.72948219592524,
2.22572782807263, 2.92360879398065, 3.75490086509486, 1.95291260429275,
2.43654883505762, 2.74020128419521, 2.19974183639896, 2.86018746220032,
3.59261460209264, 2.22390405782033, 2.90985178497341, 3.69707055350515,
2.22912572338386, 2.9241057797022, 3.74173472804349, 2.23161827669412,
2.93180030929704, 3.76745283302738, 1.96021400157099, 2.44592536388882,
2.75495164292628, 2.20602030124108, 2.87016882462401, 3.60610681143103,
2.22946798441255, 2.91882734915283, 3.71084408067234, 2.23559446368185,
2.93295016488261, 3.75708827810526, 2.23785128970986, 2.94108080663028,
3.77976388577323, 1.96856019678075, 2.46069640068571, 2.77105091902821,
2.21275319960549, 2.87997938070164, 3.61867396175126, 2.23600560246166,
2.92927064293427, 3.72598610436897, 2.24184595886197, 2.94225816715147,
3.76983808354856, 2.24596536134365, 2.9485745083463, 3.79281288014532,
1.97628019027419, 2.47210217933256, 2.78812775936112, 2.21975099503959,
2.88943946991531, 3.63386444995373, 2.24357296278298, 2.93850099550798,
3.74136265913746, 2.25014852708459, 2.9503275218798, 3.78233840784911,
2.25167806496759, 2.95787033411792, 3.80651205695025, 1.98582433124427,
2.4840140754345, 2.80696636023678, 2.22754034101027, 2.89804253908169,
3.6475807282587, 2.25145401752516, 2.94821658960953, 3.75925526443417,
2.25560261704792, 2.96084636824824, 3.79846811098198, 2.25777052793160,
2.97034997781732, 3.81888938666319, 1.99338282123118, 2.49911162022067,
2.82392759229562, 2.23584860470238, 2.91108586854350, 3.66506934920104,
2.25788606932308, 2.95824780664208, 3.77429829895792, 2.26371913853653,
2.97276652793634, 3.81277851449084, 2.26532559641099, 2.98147800797676,
3.83372594382547, 2.00352123592733, 2.51378903741398, 2.8449837855047,
2.24643285315134, 2.92335162528936, 3.6854219443556, 2.26592510857528,
2.97187662595487, 3.7898503817031, 2.27108474517556, 2.98545049021879,
3.82833777617548, 2.27275118676399, 2.99390722406367, 3.84988819397064,
2.01261514315571, 2.52759903878667, 2.86469516530926, 2.25447222697549,
2.93603541766261, 3.70113871160704, 2.27469446859609, 2.98479878169254,
3.80643968787487, 2.27872967620606, 2.99790011832985, 3.84651181684259,
2.27988920935585, 3.00529013612545, 3.86628743484644, 2.02281840175644,
2.54230191786081, 2.88840979845176, 2.26355487832797, 2.94784398937913,
3.72372570022497, 2.28180217420085, 2.99924053043030, 3.82369077242765,
2.28581023227752, 3.01101254640026, 3.86474711547053, 2.28755558085190,
3.01728703629982, 3.8840090856432, 2.03218988260106, 2.55918000535822,
2.91285045595049, 2.27330215278724, 2.96328163704208, 3.74548478489592,
2.29218107007070, 3.01354138074149, 3.84520006282845, 2.29638970067853,
3.02291289323542, 3.88543617489238, 2.29875120806278, 3.02976740738852,
3.90305033008578, 2.04365141141769, 2.57599709125068, 2.94147009954316,
2.28338923866037, 2.98056223715893, 3.76725865414008, 2.30360483959548,
3.02679737077833, 3.86792575376706, 2.30840554880696, 3.03835944621222,
3.90512119156900, 2.31018232789398, 3.04575081523367, 3.92661109582333,
2.05775961499134, 2.59563563116634, 2.97010148443806, 2.29629889010986,
2.99937796371285, 3.790024197161, 2.31424301355248, 3.04371755117878,
3.89205899556563, 2.31802794634901, 3.05228630855523, 3.93134651776682,
2.31960844363455, 3.05891983987090, 3.94850885290388, 2.07290455593282,
2.61318658431308, 2.99880071116574, 2.30968297534757, 3.0182300362712,
3.81244484420839, 2.32622597282216, 3.05823594582161, 3.91924340950534,
2.33015160144847, 3.06848780845776, 3.95426386467871, 2.33282714628515,
3.07750042555093, 3.97247127632687, 2.08573018151457, 2.63291187767,
3.03252303510393, 2.32120188794181, 3.04038284733522, 3.84358747147426,
2.34133705551313, 3.07725324893338, 3.94684568892699, 2.34413894973114,
3.09036229828436, 3.97912716731566, 2.34615425435877, 3.09711099050007,
4.00142980951883, 2.10381247226301, 2.65856886383863, 3.06745846850875,
2.33947210810481, 3.06024780786437, 3.87478305149955, 2.35451543226751,
3.10116360405558, 3.97600766790138, 2.35701974551271, 3.11081603217328,
4.01407639010318, 2.35894608057624, 3.11982681558604, 4.03492508707269,
2.12274034143787, 2.68503700708307, 3.09882391191133, 2.35476341812366,
3.08528779198250, 3.91519863316109, 2.3703340530465, 3.12653746256097,
4.01553543357259, 2.37415798162602, 3.13386847901325, 4.05306275796484,
2.37549525619605, 3.14470691965829, 4.07271715630041, 2.14594597496256,
2.72286852313628, 3.13698530915409, 2.37402027796209, 3.1159742587899,
3.95946737524246, 2.38788822898856, 3.15188312762002, 4.06046965038073,
2.38941664405227, 3.16083304895922, 4.09560460656332, 2.39125032964919,
3.16643520386011, 4.11203940618518, 2.16901291630049, 2.76045743240488,
3.18523070148978, 2.39199360336253, 3.15044469079801, 4.01368527527697,
2.40714432279085, 3.18540446077727, 4.11229022121636, 2.41044813125293,
3.19257080115555, 4.14120219480324, 2.41133838110484, 3.20077739874266,
4.15440590037256, 2.20067750079790, 2.80467421534113, 3.25840915105249,
2.41528158657702, 3.19153065337011, 4.07950269876619, 2.43114641843066,
3.22402527362553, 4.16550978799998, 2.43262481616013, 3.23394177340134,
4.19025953380752, 2.43270772115341, 3.24091911931553, 4.20860501525314,
2.23584782620152, 2.85714775503752, 3.35961532297044, 2.44577623701494,
3.24774185664101, 4.16606856488569, 2.46034450097440, 3.28059758452937,
4.24218778199947, 2.46153747539832, 3.28863254640258, 4.26303665894324,
2.46165670661943, 3.29421724800875, 4.28230990061588, 2.28952131979675,
2.92796133814304, 3.46123155558541, 2.48767372335248, 3.3196532347537,
4.2831799883297, 2.50193957388617, 3.33859742296181, 4.34328349142068,
2.50347160121869, 3.34193979691794, 4.36187000960325, 2.50348125220136,
3.34840712645623, 4.38417511435477, 2.36288190456084, 3.05951223403142,
3.61269560042025, 2.56385789438856, 3.43871361596793, 4.43963845666924,
2.57715648650667, 3.45501953729566, 4.48746892796138, 2.57724178592687,
3.45989182936763, 4.51752596826224, 2.57980707025258, 3.46016442202563,
4.53444344398332]).reshape(2, 50, 5, 3)  # max/range, level, period, h

__critval_h = np.array([0.25, 0.5, 1])
__critval_period = np.arange(2, 12, 2)
__critval_level = np.arange(0.95, 0.999, 0.001)
__critval_mr = np.array(["max", "range"])

def check(h, period, level, mr):

    if not h in __critval_h:
        raise ValueError("h can only be one of", __critval_h)

    # if not period in __critval_period:
    #    raise ValueError("period can only be one of", __critval_period)

    if not level in __critval_level:
        raise ValueError("level can only be one of", __critval_level)

    if not mr in __critval_mr:
        raise ValueError("mr can only be one of", __critval_mr)

def get_critval(h, period, level, mr):

    # Sanity check
    check(h, period, level, mr)

    index = np.zeros(4, dtype=np.int)

    # Get index into table from arguments
    index[0] = np.where(mr == __critval_mr)[0][0]
    index[1] = np.where(level == __critval_level)[0][0]
    # index[2] = np.where(period == __critval_period)[0][0]
    # print((np.abs(__critval_period - period)).argmin())
    index[2] = (np.abs(__critval_period - period)).argmin()
    index[3] = np.where(h == __critval_h)[0][0]
    # For historical reasons, the critvals are scaled by sqrt(2)
    return __critvals[tuple(index)] * np.sqrt(2)

def _find_index_date(dates, t):

    for i in range(len(dates)):
        if t < dates[i]:
            return i

    return len(dates)

def crop_data_dates(data, dates, start, end):
    """ Crops the input data and the associated
    dates w.r.t. the provided start and end
    datetime object.

    Parameters
    ----------
    data: ndarray of shape (N, W, H)
        Here, N is the number of time
        series points per pixel and W
        and H are the width and the height
        of the image, respectively.
    dates : list of datetime objects
        Specifies the dates of the elements
        in data indexed by the first axis
        n_chunks : int or None, default None
    start : datetime
        The start datetime object
    end : datetime
        The end datetime object

    Returns
    -------
    Returns: data, dates
        The cropped data array and the
        cropped list. Only those images
        and dates that are with the start/end
        period are contained in the returned
        objects.
    """

    start_idx = _find_index_date(dates, start)
    end_idx = _find_index_date(dates, end)

    data_cropped = data[start_idx:end_idx, :, :]
    dates_cropped = list(np.array(dates)[start_idx:end_idx])

    return data_cropped, dates_cropped


def compute_lam(N, hfrac, level, period):
    check(hfrac, period, 1 - level, "max")

    return get_critval(hfrac, period, 1 - level, "max")

def compute_end_history(dates, start_monitor):
    for i in range(len(dates)):
        if start_monitor <= dates[i]:
            return i

    raise Exception("Date 'start' not within the range of dates!")

def map_indices(dates):
    start = dates[0]
    end = dates[-1]
    start = datetime(start.year, 1, 1)
    end = datetime(end.year, 12, 31)

    drange = pandas.date_range(start, end, freq="d")
    ts = pandas.Series(np.ones(len(dates)), dates)
    ts = ts.reindex(drange)
    indices = np.argwhere(~np.isnan(ts).to_numpy()).T[0]

    return indices

def _nonans(xs):
  return not np.any(np.isnan(xs))

def recresid(X, y, tol=None):
    n, k = X.shape
    assert(n == y.shape[0])

    if tol is None:
        tol = np.sqrt(np.finfo(np.float32).eps) / k

    y = y.reshape(n, 1)
    ret = np.zeros(n - k)

    # initialize recursion
    yh = y[:k] # k x 1
    Xh = X[:k] # k x k
    model = sm.OLS(yh, Xh, missing="drop").fit()

    X1 = model.normalized_cov_params   # (X'X)^(-1), k x k
    bhat = np.nan_to_num(model.params) # k x 1

    check = True
    for r in range(k, n):
        # Compute recursive residual
        x = X[r]
        d = X1 @ x
        fr = 1 + (x @ d)
        resid = y[r] - np.nansum(x * bhat) # dotprod ignoring nans
        ret[r-k] = resid / np.sqrt(fr)

        # Update formulas
        X1 = X1 - (np.outer(d, d))/fr
        bhat += X1 @ x * resid

        # Check numerical stability (rectify if unstable).
        if check:
            # We check update formula value against full OLS fit
            Xh = X[:r+1]
            model = sm.OLS(y[:r+1], Xh, missing="drop").fit()

            nona = _nonans(bhat) and _nonans(model.params)
            check = not (nona and np.allclose(model.params, bhat, atol=tol))
            X1 = model.normalized_cov_params
            bhat = np.nan_to_num(model.params, 0.0)

    return ret

# From Brown, Durbin, Evans (1975).
def _pval_brownian_motion_max(x):
    Q = lambda x: 1 - stats.norm.cdf(x, loc=0, scale=1)
    p = 2 * (Q(3*x) + np.exp(-4*x**2) - np.exp(-4*x**2)*Q(x))
    return p

# CUSUM process on recursive residuals.
def rcusum(X, y):
    k, n = X.shape
    w = recresid(X.T, y)
    sigma = np.std(w)
    process = np.nancumsum(np.append([0],w))/(sigma*np.sqrt(n-k))
    return process

# Linear boundary for Brownian motion (limiting process of rec.resid. CUSUM).
def boundary(process, alpha=0.05):
    n = process.shape[0]
    lam = optimize.brentq(lambda x: _pval_brownian_motion_max(x) - alpha, 0, 20)
    t = np.linspace(0, 1, num=n)
    bounds = lam + (2*lam*t) # from Zeileis' strucchange description.
    return bounds

# Structural change test for Brownian motion.
def sctest(process):
    x = process[1:]
    n = x.shape[0]
    j = np.linspace(1/n, 1, num=n)
    x = x * 1/(1 + 2*j)
    stat = np.max(np.abs(x))
    return _pval_brownian_motion_max(stat)
