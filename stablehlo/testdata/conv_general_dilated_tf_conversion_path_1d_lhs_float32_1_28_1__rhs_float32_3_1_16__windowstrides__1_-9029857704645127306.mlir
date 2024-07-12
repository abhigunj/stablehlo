// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x28x1xf32>, tensor<3x1x16xf32>)
    %1 = call @expected() : () -> tensor<1x28x16xf32>
    %2 = stablehlo.convolution(%0#0, %0#1) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window = {pad = [[2, 2]], rhs_dilate = [2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x1xf32>, tensor<3x1x16xf32>) -> tensor<1x28x16xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1x28x16xf32>, tensor<1x28x16xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x28x1xf32>, tensor<3x1x16xf32>) {
    %0 = stablehlo.constant dense<[[[0.578237355], [-0.970109581], [-1.4660455], [-2.14128733], [-6.32983398], [3.11281967], [0.251084536], [-1.27283132], [-4.13759708], [2.18946671], [0.353660911], [-2.24349213], [-2.09625554], [2.21037674], [2.92937016], [7.15460968], [2.08107162], [0.570409536], [2.320630e+00], [1.12770963], [-0.655245662], [-0.193986982], [3.22981191], [-2.16140771], [0.855830848], [2.85118198], [-3.07397199], [1.75441694]]]> : tensor<1x28x1xf32>
    %1 = stablehlo.constant dense<[[[3.96059895, -0.153480157, 3.08699727, -4.10661936, -4.03720093, 2.35196185, -0.688390791, 0.960256397, 0.248520851, -3.88998699, -0.347455829, -0.352612913, -1.53400397, -1.73838902, 3.71750188, 1.69945824]], [[-0.827514827, 2.37026954, -1.93849611, -3.15325975, 3.96820211, -4.9135704, -2.73580456, -2.29271531, -0.932672917, 0.120558962, 3.40987921, 0.155341223, -0.659520626, 0.0879687816, -0.620618999, -2.21843314]], [[7.13872862, 2.12502098, 2.49370933, 1.45718396, 2.15042424, -1.11660945, 0.214214519, 2.22703552, 1.08319771, -4.40505075, 5.33248425, -3.68989253, -1.15562749, -0.284276783, 3.43231511, -1.942080e-01]]]> : tensor<3x1x16xf32>
    return %0, %1 : tensor<1x28x1xf32>, tensor<3x1x16xf32>
  }
  func.func private @expected() -> tensor<1x28x16xf32> {
    %0 = stablehlo.constant dense<"0x721B2FC19355DFBF90DB98C0966A7DC0A3A95BBF8A239ABFE3AFF2BFC3E692C0112608C00EE3D040FB11BBC0E0FAAF403D0BA83F236DEF3E6581ACC011817FBF8DBB67C1C130DBC072635DC008D97ABDAD4407C1BE0BE54060800C40B6D922C00213B5BF630C1541149F6BC17503F8404F51474029FC053F8EEBD7C0C259244009BC26C21C1E88C1748632C16337DFC05F1CAEC1781A7A41B76F104072E222C1A70DABC043A8CB4194D01BC2F165B74132A2EC404B4E2A3F0E5595C1B2DBAE40E035A1414819D83FAAB20E41865974411E410740E17198402A2BE6400D902E412A17A440232523C153271A41DB9F37C10C6732BFFEF71C3FEA8106419CD01F4042AD9C3FE2EA63C1D8EE054190C4D241754695C136FDDA419F0A9341589F5A4123F6B940FD5B7540ACE29DC1AE47B2BFF144C44018CEF53F58E928BF780838410925A1C1FC10A04033187DC1EB1D38C06F149241BE4797C1E111EAC0317140C1500D9AC0E9FC64418B45924034EDBD40EEF82C4035758B40892C64C138C224C173425BC2123AE7C04CC2F2C1CE629941873A8D41CB0438C17A2F324065E57DC16640C9C0AD842B42B91098C1624E8C41F43665417E3B4341DC8D17C2072C28C16C18E841D237943FD54A8C415588B2C0EC8D4EC1F01632418D79E73FAA882C41B4A28A408441AFC1E81CC840E3FC15C10FE8CEC060A9C4C0AE039F41E608F640862DDE40EC8111C107D81A41A17F4841376085C14735A441A78F3341A93D28418ABE8940C92342C0FDF544C1DA5102C0F4ACF73FE9A766BF8BE29640949718410AF3B6C152151E3F354A5CC17D469EC0160A104143F233C1CD04B3C082D033C19D3B99C03891714128C581C0271311419679464064C14240BBA75CC115B5D2C03028FDC13BCD3EC0777C95C1C6254D41129959415F0E12C18F41B73F2B3D17C1113E68C0C0F9CA41098E08C15BFB1341D3940841A63BFA40425EB6C1A617EDC08F75D2414EE574BF96F5844144E7A63FB7D14FC191475B415054A3401AB34241B4FCA040C03194C1B3105840EE6C14C15CDE8DC0353894C081F288417B4C04410461C04137DB993F845D474134D2164101985CC0C58CFB40DDCEC34069B63A414AEAA6400D8568C170990541822634C172E522C0F9E3D0BF46B94A4144D79540C5702142384CA6419D31D440A5B34A413CDB0442C602C1C1BE163EC036620B410E2FA440A92EB4C1B8DF3942521FCAC13419C9C025E10340A9846D4143B121C159268440CDFF3A414EB9DEC086DA19400D80C4418A2EADC1D502C4C06D0683C09AB77FBF54DF28BFF783AE41C282CFC06C858FBF52DA53401DF31DC0C07127C1E1FCDC405FA68E413EF2B3C05B73F6C1F48CA54139BEF4C1C2C8A7C18E2B50C1DB2EB0C0BAF923C1425CD54185E9E2BFDC4B0CC15E0458C0FF81B7402A9F43C10192D341BEA1164189BB2C41B95D73C1F503B63FAEA9BDC09ECFE6C0D46C4D40087FA63F3CEFAAC1F29F93414C5C14C171C408C16335B2C017828C41AA65B6BDE3A80F422D9F29407557BE41034BECC17F91C1C11F3F4C41DFCFC7C0CA2E01411BED1D40B7EB02C2A01FAF40EC0CD3C047794AC18D534BC150E9F041E5CA2A41DF78D23F247A7240AB5E953E5B8C86C1B11E1ABFA7D7B8C0ED7EFDC0880199C06ED816C0D5BB9DC0A2896C407BD802404ECD7DC00C8C4EC043828140BEFABDBF551071BDC1150B4049B168BF80CBC5C0C6A2E03FB1E77EC00F3E61C019111EC022618FBF763C9DBFF43627402B99303FA681B2BFA75556BF4F3A413FEB51BFBF062903424E889E40CCE783417F7830C0C4C0A0C03247A2403112633F0EC72E41D3F695406CABBAC1AFEA624157674DC17D86DBC09651A0C067F4A0414AA5984026D82CC1BB3AA7C0A62DC4BF0C68E5C039871FC1369BC040C46635BFEE4B52C05AA5F0BFE28CA3407A4349C1CD85F140C553653F9977AEBF57C846C00F1031402677573F2E321941EAC8C4C03BE3C7C01E6B8A41E6EE92C1513A03C1D81AC4C0F5E10FC031EA54BFD0DF7C41B4351BC0C54B07C00407973F525EC0BF491E07C12CFEAA416E29773F8D372B41D8443C4164CCD4BFCC5EDF402E0AD54002E7314154CBA140950C41C130D6FC40469B2CC1DF31C9BFF8D629BFB1802641BA567A4082C41DC13AFB9FC05953253F6A88A3C1710782C1395BDA40D924A7C0409CB6C0F6D054C0C3458A3F948869C11E6325410EBAFBBFF94B95C0BEC26C3F80FE8540615ECD3F6E162D415460FAC0DE4A1C40B580BE418F6AA8C151F8BDC0DD9196C028E3A5BF15FB823FEFA09E411398A8C0349E17BFB59A6040C61972C0F56C25C116DEBD401F5CEDC0009D0941F3B5C540EE737AC1BDEF8841B042FA406FD3FB40D7194540F7C86CC0C1772CC1A77F47BF78E9363F0F0CE1BFB2DBA240B161044106731D4143226E4036D2AC4044ED89C1B39091C0F610F5BF2666D8C0EB6AA4BFFE7E6DBFA3122EC1EDBB9F40C49A3BBF4DFCB0C00BAB99C0CE2A18418C12743F"> : tensor<1x28x16xf32>
    return %0 : tensor<1x28x16xf32>
  }
}

